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
    A class for detecting and tracking players with optimizations:
    1. Optical Flow tracking between detections
    2. Adaptive Resolution scaling 
    """

    def __init__(self, model_path: str,
                 device: str = "cpu",
                 frame_interval: int = 1,
                 chosen_resolution: str = "FHD",
                 onnx: bool = False,
                 input_video: str = "videos/hqsport-clip.mp4") -> None:
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
        self.n: float = 1.8
        self.p: float = 0.8
        self.default_fps: int = 25
        self.frame_rate: int = self.default_fps // self.frame_interval

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
        
        # Patch processing
        self.prev_patch_grays: dict = {}
        self.prev_patch_detections: dict = {}
        self.prev_patch_centers: dict = {}
        self.prev_gray_full_frame = None
        self.coordinates = None
        
        # 1. Adaptive Resolution
        self.adaptive_detector: AdaptiveResolutionDetector = AdaptiveResolutionDetector(self.model)
        self.last_inference_time: float = 0.0
        self.adaptive_used_this_frame: bool = False
        
        # 2. Optical Flow
        self.optical_flow: OpticalFlow = OpticalFlow()
        
        # === OPTIMIZATION SETTINGS ===
        self.use_optical_flow: bool = True
        self.use_adaptive_resolution: bool = True
        
        # Performance parameters
        self.full_detection_interval: int = frame_interval  # Run full detection every N frames
        self.max_detection_interval: int = 12  # Maximum frames between detections
        self.motion_threshold: float = 30.0  # Pixel movement threshold
        self.low_confidence_threshold: float = 0.25  # Re-detect if confidence drops

        print(f"Initialized OptimizedPlayerTracker with:")
        print(f"  Optical Flow: {self.use_optical_flow}")
        print(f"  Adaptive Resolution: {self.use_adaptive_resolution}")


    def obtain_detections(self, frame: np.ndarray) -> list[dict]:
        """
        Runs detection on a frame and returns the results.
        """
        if self.use_adaptive_resolution:
            detections, inference_time = self.adaptive_detector.detect_with_adaptive_resolution(
                frame, self.conf_thresh, self.iou_thresh
            )
            self.last_inference_time = inference_time
            self.adaptive_used_this_frame = True
            return detections

        self.adaptive_used_this_frame = False
        start_time = time.time()

        if self.onnx:
            detections = self.model.pipeline(frame, self.conf_thresh, self.iou_thresh)
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
        
        self.last_inference_time = time.time() - start_time
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


    def estimate_motion_complexity(self, gray_frame: np.ndarray, prev_gray_full_frame: np.ndarray) -> float:
        """Use the optimized motion estimation"""
        return self.optical_flow.estimate_motion_complexity(gray_frame, prev_gray_full_frame)
    

    def extract_detection_keypoints(self, detections: List[Dict]) -> List[Tuple[float, float]]:
        """Use simplified center-only approach"""
        return self.optical_flow.extract_detection_centers(detections)
    

    def propagate_detections_with_flow(self, gray_frame: np.ndarray) -> List[Dict]:
        """Use the optimized propagation method"""
        return self.optical_flow.propagate_detections(gray_frame)
    

    def process_frame_combined(
        self,
        frame: np.ndarray,
        max_patch_aspect_ratio: float = 2.0,
        overlap_ratio: float = 0.2,
        max_patches: int = 4,
        filter: bool = False
    ) -> Tuple[List[Dict], Dict]:
        """
        Combined frame processing:
        - Determines whether to split frame based on aspect ratio.
        - Uses full detection or optical flow per region.
        """
        stats = {
            "used_full_detection": False,
            "used_optical_flow": False,
            "used_adaptive_resolution": False,
            "inference_time": 0.0,
            "total_detections": 0,
            "motion_magnitude": 0.0,
            "used_patching": False,
            "num_patches": 1
        }

        h, w = frame.shape[:2]
        aspect_ratio: float = w / h
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_magnitude = self.estimate_motion_complexity(gray_frame)
        stats["motion_magnitude"] = motion_magnitude

        detections = []

        if aspect_ratio <= max_patch_aspect_ratio:
            should_full_detect = (
                self.optical_flow.prev_gray is None or
                not self.optical_flow.prev_detections or
                self.frames_since_full_detection >= self.max_detection_interval or
                (self.frames_since_full_detection >= self.full_detection_interval and
                motion_magnitude > self.motion_threshold)
            )

            if should_full_detect:
                detections = self.obtain_detections(frame)
                stats["inference_time"] = self.last_inference_time
                stats["used_adaptive_resolution"] = self.adaptive_used_this_frame
                stats["used_full_detection"] = True
                self.frames_since_full_detection = 0

                self.optical_flow.prev_detections = detections.copy()
                self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)

            else:
                detections = self.optical_flow.propagate_detections(gray_frame)
                if detections:
                    stats["used_optical_flow"] = True
                    self.frames_since_full_detection += 1
                else:
                    detections = self.obtain_detections(frame)
                    stats["inference_time"] = self.last_inference_time
                    stats["used_adaptive_resolution"] = self.adaptive_used_this_frame
                    stats["used_full_detection"] = True
                    self.frames_since_full_detection = 0

                    self.optical_flow.prev_detections = detections.copy()
                    self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)

        else:
            stats["used_patching"] = True
            target_patch_ratio: float = 1.2
            initial_patch_w: int = int(h * target_patch_ratio)
            effective_patch_w = max(initial_patch_w, int(w / max_patches))
            step = int(effective_patch_w * (1 - overlap_ratio))

            coords = []
            x = 0
            while x + effective_patch_w < w:
                coords.append(x)
                x += step

            if not coords or coords[-1] + effective_patch_w < w:
                coords.append(w - effective_patch_w)

            if len(coords) > max_patches:
                step = (w - effective_patch_w) // (max_patches - 1)
                coords = [i * step for i in range(max_patches - 1)] + [w - effective_patch_w]

            all_detections = []
            stats["num_patches"] = len(coords)

            for x_offset in coords:
                patch = frame[:, x_offset:x_offset + effective_patch_w]
                patch_gray = gray_frame[:, x_offset:x_offset + effective_patch_w]

                should_full_detect = (
                    self.optical_flow.prev_gray is None or
                    not self.optical_flow.prev_detections or
                    self.frames_since_full_detection >= self.max_detection_interval or
                    (self.frames_since_full_detection >= self.full_detection_interval and
                    motion_magnitude > self.motion_threshold)
                )

                if should_full_detect:
                    patch_detections = self.obtain_detections(patch)
                    stats["used_full_detection"] = True
                    stats["inference_time"] += self.last_inference_time
                    stats["used_adaptive_resolution"] |= self.adaptive_used_this_frame
                    self.frames_since_full_detection = 0

                    self.optical_flow.prev_detections = patch_detections.copy()
                    self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(patch_detections)
                else:
                    patch_detections = self.optical_flow.propagate_detections(patch_gray)
                    if patch_detections:
                        stats["used_optical_flow"] = True
                        self.frames_since_full_detection += 1
                    else:
                        patch_detections = self.obtain_detections(patch)
                        stats["used_full_detection"] = True
                        stats["inference_time"] += self.last_inference_time
                        stats["used_adaptive_resolution"] |= self.adaptive_used_this_frame
                        self.frames_since_full_detection = 0

                        self.optical_flow.prev_detections = patch_detections.copy()
                        self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(patch_detections)

                for d in patch_detections:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = (x1 + x_offset, y1, x2 + x_offset, y2)
                    all_detections.append(d)

            detections = non_max_merge(all_detections, iou_thresh=0.5)

        if filter and self.field_mask is not None:
            detections = filter_detections_by_mask(detections, self.field_mask)

        self.optical_flow.prev_gray = gray_frame.copy()
        stats["total_detections"] = len(detections)
        return detections, stats



    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Simplified optimization focusing only on optical flow
        """
        stats = {
            "used_full_detection": False,
            "used_optical_flow": False,
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
            detections = self.obtain_detections(frame)
            stats["inference_time"] = self.last_inference_time
            stats["used_adaptive_resolution"] = self.adaptive_used_this_frame

            
            # Update optical flow state
            self.optical_flow.prev_detections = detections.copy()
            self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)
            
            self.frames_since_full_detection = 0
            stats["used_full_detection"] = True
            
        else:
            detections = self.optical_flow.propagate_detections(gray_frame)
            
            if detections:
                stats["used_optical_flow"] = False
                self.frames_since_full_detection += 1
            else:
                # Fallback to full detection if propagation fails
                detections = self.obtain_detections(frame)
                stats["inference_time"] = self.last_inference_time
                stats["used_adaptive_resolution"] = self.adaptive_used_this_frame

                stats["used_full_detection"] = True
                self.frames_since_full_detection = 0
                
                # Update optical flow state
                self.optical_flow.prev_detections = detections.copy()
                self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)
        
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
            target_patch_ratio: float = 1.2
            initial_patch_w: int = int(h * target_patch_ratio)

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
    

    def process_frame_combined_with_patch_flow(
        self,
        frame: np.ndarray,
        max_patch_aspect_ratio: float = 2.0,
        overlap_ratio: float = 0.2,
        max_patches: int = 3,
        filter: bool = False
    ) -> Tuple[List[Dict], Dict]:
        """
        Combined processing with patch-aware optical flow tracking.
        Maintains per-patch grayscale history to support optical flow.
        """
        stats = {
            "used_full_detection": False,
            "used_optical_flow": False,
            "used_adaptive_resolution": False,
            "inference_time": 0.0,
            "total_detections": 0,
            "motion_magnitude": 0.0,
            "used_patching": False,
            "num_patches": 1
        }

        if not hasattr(self, "prev_patch_grays"):
            self.prev_patch_grays = {}
            self.prev_patch_detections = {}
            self.prev_patch_centers = {}

        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_magnitude = self.estimate_motion_complexity(gray_frame, self.prev_gray_full_frame)
        stats["motion_magnitude"] = motion_magnitude

        aspect_ratio = w / h
        all_detections = []

        if aspect_ratio <= max_patch_aspect_ratio:
            # No patching, single full-frame detection with global optical flow
            patch = frame
            patch_gray = gray_frame
            patch_idx = 0

            should_full_detect = (
                self.optical_flow.prev_gray is None or
                not self.optical_flow.prev_detections or
                self.frames_since_full_detection >= self.max_detection_interval or
                (self.frames_since_full_detection >= self.full_detection_interval and 
                motion_magnitude > self.motion_threshold)
            )

            if should_full_detect:
                detections = self.obtain_detections(patch)
                stats["used_full_detection"] = True
                stats["inference_time"] = self.last_inference_time
                stats["used_adaptive_resolution"] = self.adaptive_used_this_frame
                self.frames_since_full_detection = 0

                self.optical_flow.prev_detections = detections.copy()
                self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)
                self.optical_flow.prev_gray = patch_gray.copy()
            else:
                detections = self.optical_flow.propagate_detections(patch_gray)
                if detections:
                    stats["used_optical_flow"] = True
                    self.frames_since_full_detection += 1
                else:
                    detections = self.obtain_detections(patch)
                    stats["used_full_detection"] = True
                    stats["inference_time"] = self.last_inference_time
                    stats["used_adaptive_resolution"] = self.adaptive_used_this_frame
                    self.frames_since_full_detection = 0

                    self.optical_flow.prev_detections = detections.copy()
                    self.optical_flow.detection_centers = self.optical_flow.extract_detection_centers(detections)
                    self.optical_flow.prev_gray = patch_gray.copy()

            all_detections = detections

        else:
            # Patching with per-patch flow
            stats["used_patching"] = True
            if self.coordinates is None:
                target_patch_ratio = 1.2
                initial_patch_w = int(h * target_patch_ratio)
                effective_patch_w = max(initial_patch_w, int(w / max_patches))
                step = int(effective_patch_w * (1 - overlap_ratio))

                coords = []
                x = 0
                while x + effective_patch_w < w:
                    coords.append(x)
                    x += step

                # Ensure the final patch covers the right edge
                if not coords or coords[-1] + effective_patch_w < w:
                    coords.append(w - effective_patch_w)

                # Enforce patch limit
                if max_patches is not None and len(coords) > max_patches:
                    step = (w - effective_patch_w) // (max_patches - 1)
                    coords = [i * step for i in range(max_patches - 1)] + [w - effective_patch_w]

                self.coordinates = coords  # Cache for future frames
                self.effective_patch_w = effective_patch_w

            stats["num_patches"] = len(self.coordinates)

            for patch_idx, x_offset in enumerate(self.coordinates):
                patch = frame[:, x_offset:x_offset + self.effective_patch_w]
                patch_gray = gray_frame[:, x_offset:x_offset + self.effective_patch_w]


                prev_gray = self.prev_patch_grays.get(patch_idx)
                prev_detections = self.prev_patch_detections.get(patch_idx, [])
                prev_centers = self.prev_patch_centers.get(patch_idx, [])

                should_full_detect = (
                    prev_gray is None or
                    not prev_detections or
                    self.frames_since_full_detection >= self.max_detection_interval or
                    (self.frames_since_full_detection >= self.full_detection_interval and 
                    motion_magnitude > self.motion_threshold)
                )

                if should_full_detect:
                    patch_detections = self.obtain_detections(patch)
                    stats["used_full_detection"] = True
                    stats["inference_time"] += self.last_inference_time
                    stats["used_adaptive_resolution"] |= self.adaptive_used_this_frame
                    self.frames_since_full_detection = 0
                else:
                    # Temporarily inject per-patch data for optical flow
                    self.optical_flow.prev_gray = prev_gray
                    self.optical_flow.prev_detections = prev_detections
                    self.optical_flow.detection_centers = prev_centers

                    patch_detections = self.optical_flow.propagate_detections(patch_gray)
                    if patch_detections:
                        stats["used_optical_flow"] = True
                        self.frames_since_full_detection += 1
                    else:
                        patch_detections = self.obtain_detections(patch)
                        stats["used_full_detection"] = True
                        stats["inference_time"] += self.last_inference_time
                        stats["used_adaptive_resolution"] |= self.adaptive_used_this_frame
                        self.frames_since_full_detection = 0

                relative_detections = []
                for d in patch_detections:
                    x1, y1, x2, y2 = d["bbox"]
                    rel_box = (x1 - x_offset, y1, x2 - x_offset, y2)
                    relative_detections.append({
                        "bbox": rel_box,
                        "conf": d["conf"],
                        "class": d["class"],
                        "propagated": d.get("propagated", False)
                    })

                self.prev_patch_grays[patch_idx] = patch_gray.copy()
                self.prev_patch_detections[patch_idx] = relative_detections
                self.prev_patch_centers[patch_idx] = self.optical_flow.extract_detection_centers(relative_detections)

                # Offset detection coordinates to global frame
                for d in patch_detections:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = (x1 + x_offset, y1, x2 + x_offset, y2)
                    all_detections.append(d)

            all_detections = non_max_merge(all_detections, iou_thresh=0.5)

        if filter and self.field_mask is not None:
            all_detections = filter_detections_by_mask(all_detections, self.field_mask)
        self.prev_gray_full_frame = gray_frame.copy()

        stats["total_detections"] = len(all_detections)
        return all_detections, stats


    def process_video(
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
            "adaptive_resolution_frames": 0,
            "total_inference_time": 0.0,
            "motion_magnitudes": []
        }

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
            detections, frame_stats = self.process_frame_combined_with_patch_flow(frame)

            # Update statistics
            if frame_stats["used_full_detection"]:
                optimization_stats["full_detections"] += 1
            if frame_stats["used_optical_flow"]:
                optimization_stats["optical_flow_frames"] += 1
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
    tracker.use_adaptive_resolution = False
    
    # Optimize parameters for performance
    tracker.max_detection_interval = 12  # Force detection after 12 frames
    tracker.conf_thresh = 0.1  # Slightly lower confidence for better tracking
    tracker.motion_threshold = 5.0  # Adjusted for sports videos
    
    return tracker


if __name__ == "__main__":
    # Example usage
    input_video = "videos/Futbol_Ejemplo_ Wide_1.mp4"
    output_dir = "videos/output_videos" 
    os.makedirs(output_dir, exist_ok=True)

    # model_path: str = "models/yolov8n.pt"
    model_path: str = "models/yolo8n_static_quantized.onnx"

    onnx: bool = ".onnx" in model_path

    # Create optimized tracker with all features enabled
    tracker: OptimizedPlayerTracker = create_optimized_tracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        frame_interval=3,
        onnx=onnx,
        chosen_resolution="FHD",
        input_video=input_video
    )

    output_path = os.path.join(output_dir, "output_optimized_tracking.mp4")
    
    print(f"Processing video with optimized tracker...")
    tracker.process_video(
        input_video,
        output_path,
        tracking=False,
        display=True,
        real_output=False
    )