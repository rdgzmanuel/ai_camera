import os
import time
from collections import deque
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker, STrack
from ultralytics import YOLO
from src.quantization.onnx_detector import OnnxDetector
from src.adaptive_resolution import AdaptiveResolutionDetector
from src.predictive_roi_processor import PredictiveROIProcessor
from src.optical_flow import OpticalFlow
from src.controller import BoxController
from src.utils import (
    get_boxes,
    compute_fps,
    draw_tracks,
    tracks_to_detections,
    draw_boxes,
    draw_detections,
    non_max_merge,
    load_field_mask,
    filter_detections_by_mask,
    compute_target_box
)


class OptimizedPlayerTracker:
    """
    A class for detecting and tracking players with all three optimizations:
    1. Optical Flow tracking between detections
    2. Adaptive Resolution scaling 
    3. Smart ROI processing
    """

    def __init__(self, model_path: str,
                 device: str = "cpu",
                 frame_interval: int = 1,
                 chosen_resolution: str = "FHD",
                 onnx: bool = False,
                 input_video: str = "videos/hqsport-clip-3.mp4") -> None:
        """
        Initializes the OptimizedPlayerTracker.
        """
        self.device: str = device
        self.conf_thresh: float = 0.3
        self.iou_thresh: float = 0.1
        self.max_det: int = 300
        self.onnx: bool = onnx

        if self.onnx:
            self.model: OnnxDetector = OnnxDetector(model_path)
        else:
            self.model: YOLO = YOLO(model_path).to(device)

        self.frame_interval: int = frame_interval
        self.n: float = 1.8 # Scale factor for expanded box
        self.p: float = 0.8  # Percentage of boxes to consider
        self.default_fps: int = 25
        self.frame_rate: int = self.default_fps // self.frame_interval
        self.full_frame_interval_factor: int = self.default_fps // (self.frame_interval + 2)

        tracker_args: dict = {
            "track_thresh": self.conf_thresh,
            "track_buffer": 30,
            "match_thresh": 0.99,
            "min_box_area": 5,
            "mot20": False,
        }
        self.tracker: BYTETracker = BYTETracker(SimpleNamespace(**tracker_args), frame_rate=self.frame_rate)

        # Visualization parameters
        self.aspect_ratio: tuple[int, int] = (16, 9)
        self.target_ratio: float = self.aspect_ratio[0] / self.aspect_ratio[1]
        self.current_box_info: Optional[tuple[float, float, float, float]] = None

        self.resolutions: dict[str, tuple[int, int]] = {
            "HD": (1280, 720),
            "FHD": (1920, 1080),
            "QHD": (2560, 1440),
            "4K": (3840, 2160),
        }
        self.resolution: tuple[int, int] = self.resolutions[chosen_resolution]
        
        # Frame optimization
        self.expanded_box: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
        self.input_resolution: Optional[tuple[int, int]] = None  # Default model input size

        # PID controller
        self.box_controller: BoxController = BoxController(target_ratio=self.target_ratio)

        # Field detection
        self.field_mask: Optional[np.ndarray] = None
        input_video: str = input_video.split("/")[-1][:-4]
        self.mask_path: str = f"field_coordinates/{input_video}.json"
        
        # === OPTIMIZATION COMPONENTS ===
        
        # 1. Adaptive Resolution
        self.adaptive_detector = AdaptiveResolutionDetector(self.model)
        
        # 2. Smart ROI
        self.roi_processor = None
        
        # 3. Optical Flow
        self.optical_flow: OpticalFlow = OpticalFlow()
        
        # === OPTIMIZATION SETTINGS ===
        self.use_optical_flow = True
        self.use_smart_roi = False
        self.use_adaptive_resolution = False
        
        # Performance parameters
        self.full_detection_interval = 5  # Run full detection every N frames
        self.max_detection_interval = 12  # Maximum frames between detections
        self.motion_threshold = 30.0  # Pixel movement threshold
        self.low_confidence_threshold = 0.25  # Re-detect if confidence drops

        print(f"Initialized OptimizedPlayerTracker with:")
        print(f"  Optical Flow: {self.use_optical_flow}")
        print(f"  Smart ROI: {self.use_smart_roi}")
        print(f"  Adaptive Resolution: {self.use_adaptive_resolution}")


    def obtain_detections(self, frame: np.ndarray) -> list[dict]:
        """
        Runs detection on a frame and returns the results.
        """
        if self.onnx:
            detections: list[dict] = self.model.pipeline(frame, self.conf_thresh, self.iou_thresh)

        else:
            results = self.model(
                frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                verbose=False
            )[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf >= self.conf_thresh and cls == 0:
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "class": cls
                    })
        return detections


    def obtain_tracks(self, frame: np.ndarray, detections: list[dict]) -> list[STrack]:
        """
        Runs tracking on a frame using the provided detections.
        """
        h, w = frame.shape[:2]

        tracking_input = np.array(
            [[*det["bbox"], det["conf"]] for det in detections],
            dtype=np.float32
        ) if detections else np.empty((0, 5), dtype=np.float32)

        tracks: list[STrack] = []
        if tracking_input.size > 0:
            tracks = self.tracker.update(tracking_input, [h, w], [h, w])

        return tracks

    # === OPTICAL FLOW METHODS ===
    def estimate_motion_complexity(self, gray_frame: np.ndarray) -> float:
        """Use the optimized motion estimation"""
        return self.optical_flow.estimate_motion_complexity_fast(gray_frame)
    

    def extract_detection_keypoints(self, detections: List[Dict]) -> List[Tuple[float, float]]:
        """Use simplified center-only approach"""
        return self.optical_flow.extract_detection_centers(detections)
    

    def propagate_detections_with_flow(self, gray_frame: np.ndarray) -> List[Dict]:
        """Use the optimized propagation method"""
        return self.optical_flow.propagate_detections_simple(gray_frame)


    # === SMART ROI METHODS ===
    
    def process_frame_with_smart_roi(self, frame: np.ndarray) -> List[Dict]:
        """
        Process frame using smart ROI with previous detection context.
        """
        if not hasattr(self, 'roi_processor') or self.roi_processor is None:
            return self.obtain_detections(frame)
        
        # Use previous detections to create focused ROIs
        predicted_boxes = []
        if self.prev_detections:
            for det in self.prev_detections:
                x1, y1, x2, y2 = det["bbox"]
                # Expand the box slightly for ROI
                w, h = x2 - x1, y2 - y1
                expansion = 0.3
                exp_x1 = max(0, int(x1 - w * expansion))
                exp_y1 = max(0, int(y1 - h * expansion))
                exp_x2 = min(frame.shape[1], int(x2 + w * expansion))
                exp_y2 = min(frame.shape[0], int(y2 + h * expansion))
                predicted_boxes.append((exp_x1, exp_y1, exp_x2, exp_y2))
        
        # Create ROIs
        smart_rois = self.roi_processor.cluster_boxes_to_rois(predicted_boxes, max_rois=3)
        
        all_detections = []
        
        # Process each ROI
        for roi_x1, roi_y1, roi_x2, roi_y2 in smart_rois:
            # Ensure ROI is valid
            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                continue
                
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_detections = self.obtain_detections(roi)
            
            # Adjust coordinates back to full frame
            for det in roi_detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1)
                all_detections.append(det)
        
        # If no ROIs or no detections, fall back to full frame with lower resolution
        if not all_detections:
            # Quick full-frame detection at lower resolution
            h, w = frame.shape[:2]
            scale = 0.5
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            detections = self.obtain_detections(small_frame)
            
            # Scale back up
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (x1/scale, y1/scale, x2/scale, y2/scale)
                all_detections.append(det)
        
        return all_detections

    # === COMBINED OPTIMIZATION METHOD ===
    def process_frame_fully_optimized(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Simplified optimization focusing only on optical flow
        """
        stats = {
            "used_full_detection": False,
            "used_optical_flow": False,
            "used_smart_roi": False,
            "used_adaptive_resolution": False,
            "inference_time": 0.0,
            "total_detections": 0,
            "motion_magnitude": 0.0
        }
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fast motion estimation
        motion_magnitude = self.estimate_motion_complexity(gray_frame)
        stats["motion_magnitude"] = motion_magnitude
        
        # Simplified decision logic
        should_full_detect = (
            self.optical_flow.prev_gray is None or  # First frame
            not self.optical_flow.prev_detections or  # No previous detections
            self.frames_since_full_detection >= self.max_detection_interval or  # Too long since detection
            (self.frames_since_full_detection >= self.full_detection_interval and 
             motion_magnitude > self.motion_threshold)  # High motion
        )
        
        if should_full_detect:
            # Full YOLO detection
            start_time = time.time()
            detections = self.obtain_detections(frame)
            stats["inference_time"] = time.time() - start_time
            
            # Update optical flow state
            self.optical_flow.prev_detections = detections.copy()
            self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)
            
            self.frames_since_full_detection = 0
            stats["used_full_detection"] = True
            
        else:
            # Try optical flow propagation
            detections = self.optical_flow.propagate_detections_simple(gray_frame)
            
            if detections:
                stats["used_optical_flow"] = True
            else:
                # Fallback to full detection if propagation fails
                start_time = time.time()
                detections = self.obtain_detections(frame)
                stats["inference_time"] = time.time() - start_time
                stats["used_full_detection"] = True
                self.frames_since_full_detection = 0
                
                # Update optical flow state
                self.optical_flow.prev_detections = detections.copy()
                self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)
            
            self.frames_since_full_detection += 1
        
        # Update previous frame for next iteration
        self.optical_flow.prev_gray = gray_frame.copy()
        
        stats["total_detections"] = len(detections)
        return detections, stats


    def update_current_box(self, target: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return self.box_controller.update(target)


    def process_frame_patched(
        self,
        frame: np.ndarray,
        max_patch_aspect_ratio: float = 2.0,
        overlap_ratio: float = 0.2,
        max_patches: int = 4,
        filter=False
    ) -> list[dict]:
        """
        Process frame by checking its aspect ratio and optionally splitting into overlapping patches.
        """
        h, w = frame.shape[:2]
        aspect_ratio: float = w / h

        if aspect_ratio <= max_patch_aspect_ratio:
            detections = self.obtain_detections(frame)
        
        else:
            # Calculate initial patch width based on near-square target ratio
            target_patch_ratio = 1.2
            initial_patch_w = int(h * target_patch_ratio)

            # Recompute patch width to not exceed max_patches
            effective_patch_w = max(initial_patch_w, int(w / max_patches))
            step = int(effective_patch_w * (1 - overlap_ratio))

            coords = []
            x = 0
            while x + effective_patch_w < w:
                coords.append(x)
                x += step

            # Always ensure final patch covers right edge
            if not coords or coords[-1] + effective_patch_w < w:
                coords.append(w - effective_patch_w)

            # Truncate to max_patches if necessary
            if len(coords) > max_patches:
                step = (w - effective_patch_w) // (max_patches - 1)
                coords = [i * step for i in range(max_patches - 1)] + [w - effective_patch_w]

            all_detections = []
            for x_offset in coords:
                patch = frame[:, x_offset:x_offset + effective_patch_w]
                detections = self.obtain_detections(patch)

                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = (x1 + x_offset, y1, x2 + x_offset, y2)
                    all_detections.append(d)
            
            detections: list[dict] = non_max_merge(all_detections, iou_thresh=0.5)

        if filter and self.field_mask is not None:
            detections = filter_detections_by_mask(detections, self.field_mask)

        return detections

    # === MAIN PROCESSING METHOD ===
    
    def process_video_optimized(
        self,
        input_path: str,
        output_path: str,
        tracking: bool = True,
        display: bool = False,
        real_output: bool = False
    ) -> None:
        """
        Process video with all optimizations enabled.
        """
        cap = cv2.VideoCapture(input_path)
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_id: int = 0
        fps_window: deque = deque(maxlen=30)
        total_start_time: float = time.time()
        total_frames: int = 0
        target_box: Optional[tuple[float, float, float, float]] = None

        # Optimization statistics
        optimization_stats: dict = {
            "full_detections": 0,
            "optical_flow_frames": 0,
            "smart_roi_frames": 0,
            "adaptive_resolution_frames": 0,
            "total_inference_time": 0.0,
            "motion_magnitudes": []
        }

        print(f"Starting optimized video processing...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Optimizations enabled:")
        print(f" - Optical Flow: {self.use_optical_flow}")
        print(f" - Smart ROI: {self.use_smart_roi}")
        print(f" - Adaptive Resolution: {self.use_adaptive_resolution}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.input_resolution is None:
                self.input_resolution = (frame.shape[1], frame.shape[0])

            # Load field mask once
            if self.field_mask is None and os.path.exists(self.mask_path):
                self.field_mask = load_field_mask(self.mask_path, (frame.shape[1], frame.shape[0]))

            frame_start: float = time.time()

            # Use fully optimized detection
            detections, frame_stats = self.process_frame_fully_optimized(frame)

            # Update statistics
            if frame_stats["used_full_detection"]:
                optimization_stats["full_detections"] += 1
            if frame_stats["used_optical_flow"]:
                optimization_stats["optical_flow_frames"] += 1
            if frame_stats["used_smart_roi"]:
                optimization_stats["smart_roi_frames"] += 1
            if frame_stats["used_adaptive_resolution"]:
                optimization_stats["adaptive_resolution_frames"] += 1
            
            optimization_stats["total_inference_time"] += frame_stats["inference_time"]
            optimization_stats["motion_magnitudes"].append(frame_stats["motion_magnitude"])

            # Apply field mask filtering if available
            if self.field_mask is not None and detections:
                detections = filter_detections_by_mask(detections, self.field_mask)

            # Tracking and focus box computation
            tracks = []
            if tracking and detections:
                tracks = self.obtain_tracks(frame, detections)
                focus_input = tracks_to_detections(tracks) if tracks else detections
            else:
                focus_input = detections

            if focus_input:
                target_box = compute_target_box(focus_input, self.p, self.target_ratio)

            # Update ROI processor with tracks
            if self.roi_processor and tracks:
                self.roi_processor.update_player_velocities(tracks)

            # Draw results
            if target_box is not None:
                current_box = self.update_current_box(target_box)
                tight_box, expanded_box = get_boxes(current_box, self.n, frame.shape, self.target_ratio)
                self.expanded_box = expanded_box

                if real_output:
                    ex1, ey1, ex2, ey2 = expanded_box
                    frame = frame[ey1:ey2, ex1:ex2]
                else:
                    if tracking and tracks:
                        frame = draw_tracks(frame, tracks)
                    else:
                        frame = draw_detections(frame, detections)
                    frame = draw_boxes(frame, tight_box, expanded_box)

            frame = cv2.resize(frame, self.resolution)

            frame, effective_fps, total_frames = compute_fps(
                frame, fps_window, total_start_time, total_frames, frame_start
            )

            # Add optimization info overlay
            info_lines = [
                f"FPS: {effective_fps:.1f}",
                f"Detections: {len(detections)}",
                f"Motion: {frame_stats['motion_magnitude']:.1f}",
            ]
            
            # Show current mode
            if frame_stats["used_full_detection"]:
                if frame_stats["used_adaptive_resolution"]:
                    info_lines.append(f"Mode: Full+Adaptive ({self.adaptive_detector.current_size})")
                else:
                    info_lines.append("Mode: Full Detection")
            elif frame_stats["used_optical_flow"]:
                info_lines.append("Mode: Optical Flow")
            elif frame_stats["used_smart_roi"]:
                info_lines.append("Mode: Smart ROI")
            else:
                info_lines.append("Mode: Standard")

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out.write(frame)

            if display:
                cv2.imshow("Optimized Player Tracking", frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    break

            if frame_id % 30 == 0:
                total_processed = frame_id + 1
                compute_savings = (optimization_stats["optical_flow_frames"] / max(1, total_processed)) * 100
                avg_motion = np.mean(optimization_stats["motion_magnitudes"][-30:]) if optimization_stats["motion_magnitudes"] else 0
                
                if optimization_stats["full_detections"] > 0:
                    avg_inference = optimization_stats["total_inference_time"] / optimization_stats["full_detections"]
                    print(f"Frame {frame_id} | FPS: {effective_fps:.1f} | "
                          f"Compute savings: {compute_savings:.1f}% | "
                          f"Avg inference: {avg_inference*1000:.1f}ms | "
                          f"Motion: {avg_motion:.1f}")
                else:
                    print(f"Frame {frame_id} | FPS: {effective_fps:.1f} | "
                          f"Compute savings: {compute_savings:.1f}% | "
                          f"Motion: {avg_motion:.1f}")

            frame_id += 1

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - total_start_time
        total_processed = frame_id
        
        compute_savings = (optimization_stats["optical_flow_frames"] / max(1, total_processed)) * 100
        roi_usage = (optimization_stats["smart_roi_frames"] / max(1, total_processed)) * 100
        adaptive_usage = (optimization_stats["adaptive_resolution_frames"] / max(1, total_processed)) * 100

        print(f"\n{'='*50}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*50}")
        print(f"Total frames processed: {total_processed}")
        print(f"Processing time: {total_time:.1f}s")
        print(f"Average FPS: {total_processed/total_time:.1f}")
        print(f"")
        print(f"Detection Method Usage:")
        print(f"  Full detections: {optimization_stats['full_detections']} ({optimization_stats['full_detections']/total_processed*100:.1f}%)")
        print(f"  Optical flow frames: {optimization_stats['optical_flow_frames']} ({compute_savings:.1f}%)")
        print(f"  Smart ROI frames: {optimization_stats['smart_roi_frames']} ({roi_usage:.1f}%)")
        print(f"")
        print(f"Feature Usage:")
        print(f"  Adaptive resolution: {optimization_stats['adaptive_resolution_frames']} frames ({adaptive_usage:.1f}%)")
        
        if optimization_stats["full_detections"] > 0:
            avg_inference = optimization_stats["total_inference_time"] / optimization_stats["full_detections"]
            print(f"  Average inference time: {avg_inference*1000:.1f}ms per detection")
        
        if optimization_stats["motion_magnitudes"]:
            avg_motion = np.mean(optimization_stats["motion_magnitudes"])
            print(f"  Average scene motion: {avg_motion:.1f} pixels/frame")
        
        print(f"")
        print(f"Performance Improvement:")
        theoretical_standard_time = total_processed * 0.08  # Assume 80ms per frame for standard processing
        actual_time = total_time
        speedup = theoretical_standard_time / actual_time if actual_time > 0 else 1
        print(f"  Estimated speedup: {speedup:.1f}x")
        print(f"  Computational savings: {compute_savings:.1f}%")


def create_optimized_tracker(model_path: str, **kwargs) -> OptimizedPlayerTracker:
    """
    Factory function to create a fully optimized tracker with recommended settings.
    """
    tracker = OptimizedPlayerTracker(
        model_path=model_path,
        **kwargs
    )
    
    # Enable all optimizations by default
    tracker.use_optical_flow = True
    tracker.use_smart_roi = False  
    tracker.use_adaptive_resolution = False
    
    # Optimize parameters for performance
    tracker.full_detection_interval = 6  # Run full detection every 6 frames
    tracker.max_detection_interval = 12  # Force detection after 12 frames
    tracker.conf_thresh = 0.25  # Slightly lower confidence for better tracking
    tracker.motion_threshold = 35.0  # Adjusted for sports videos
    
    return tracker


if __name__ == "__main__":
    # Example usage
    input_video = "videos/hqsport-clip.mp4"
    output_dir = "videos/output_videos" 
    os.makedirs(output_dir, exist_ok=True)

    # model_path: str = "models/yolov8n.pt"
    model_path: str = "models/yolo8n_static_quantized.onnx"

    onnx: bool = ".onnx" in model_path

    # Create optimized tracker with all features enabled
    tracker: OptimizedPlayerTracker = create_optimized_tracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        frame_interval=3,  # Can be 1 with optimizations
        onnx=onnx,
        chosen_resolution="FHD",
        input_video=input_video
    )

    output_path = os.path.join(output_dir, "output_optimized_tracking.mp4")
    
    print(f"Processing video with optimized tracker...")
    tracker.process_video_optimized(
        input_video,
        output_path,
        tracking=False,
        display=True,
        real_output=True
    )