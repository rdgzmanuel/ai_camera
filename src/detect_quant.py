import os
import time
from collections import deque
from typing import Optional
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker, STrack
from ultralytics import YOLO
from src.quantization.onnx_detector import OnnxDetector
from src.controller import BoxController
from src.utils import (
    enforce_aspect_ratio,
    compute_fps,
    draw_tracks,
    tracks_to_detections,
    draw_boxes,
    non_max_merge,
    load_field_mask,
    filter_detections_by_mask
)


class PlayerTracker:
    """
    A class for detecting and optionally tracking moving objects in video streams.
    Computes dynamic focus boxes and smooth camera movement.
    """

    def __init__(self, model_path: str,
                 device: str = "cpu",
                 frame_interval: int = 1,
                 chosen_resolution: str = "FHD",
                 onnx: bool = False,
                 input_video: str = "videos/hqsport-clip-3.mp4") -> None:
        """
        Initializes the PlayerTracker.

        Args:
            model_path: Path to YOLO model.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.device: str = device
        self.conf_thresh: float = 0.15
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


    def process_video(
        self,
        input_path: str,
        output_path: str,
        tracking: bool = True,
        display: bool = False,
        real_output: bool = False
    ) -> None:
        """
        Processes a video file for detection and tracking, and writes output video.

        Args:
            input_path: Path to the input video.
            output_path: Path to save the output video.
            tracking: Whether to enable tracking.
            display: Whether to show live display window.
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

        while True:
            ret, frame = cap.read()
            if self.input_resolution is None:
                self.input_resolution = (frame.shape[1], frame.shape[0])
            if not ret:
                break

            if self.field_mask is None:
                # Load field mask once
                self.field_mask = load_field_mask(self.mask_path, (frame.shape[1], frame.shape[0]))

            frame_start: float = time.time()

            if frame_id % self.frame_interval == 0:
                # Full detection every n * frame_interval frames
                if frame_id % (self.frame_interval * self.full_frame_interval_factor) == 0 or self.expanded_box is None:
                    detections = self.process_frame_patched(frame, filter=True)

                else:
                    # ROI detection inside expanded box
                    ex1, ey1, ex2, ey2 = self.expanded_box
                    roi = frame[ey1:ey2, ex1:ex2]
                    detections = self.process_frame_patched(roi)
                    for d in detections:
                        x1, y1, x2, y2 = d["bbox"]
                        d["bbox"] = (int(x1 + ex1), int(y1 + ey1), int(x2 + ex1), int(y2 + ey1))

                # Update focus box
                if tracking:
                    tracks = self.obtain_tracks(frame, detections)
                    focus_input = tracks_to_detections(tracks)
                else:
                    focus_input = detections

                target_box = self.compute_target_box(focus_input, self.p)

            if target_box is not None:
                current_box = self.update_current_box(target_box)
                tight_box, expanded_box = self.get_boxes(current_box, self.n, frame.shape)

                self.expanded_box = expanded_box

                if real_output:
                    # Crop to expanded box only
                    ex1, ey1, ex2, ey2 = expanded_box
                    frame = frame[ey1:ey2, ex1:ex2]

                    # Optionally, resize to original resolution so all outputs are same size
                else:
                    if tracking:
                        frame = draw_tracks(frame, tracks)
                    else:
                        frame = self.draw_detections(frame, detections)

                    frame = draw_boxes(frame, tight_box, expanded_box)
            frame = cv2.resize(frame, self.resolution)

            frame, effective_fps, total_frames = compute_fps(
                frame, fps_window, total_start_time, total_frames, frame_start
            )

            out.write(frame)

            if display:
                cv2.imshow("Tracking" if tracking else "Detection", frame)

                if cv2.waitKey(1) == 27:
                    break

            if frame_id % 10 == 0:
                print(
                    f"Frame {frame_id} | Overall FPS: {effective_fps:.2f}"
                )
            frame_id += 1

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        total_elapsed = time.time() - total_start_time
        final_fps = total_frames / total_elapsed
        print(f"Processed {total_frames} frames in {total_elapsed:.2f}s | Effective FPS: {final_fps:.2f}")
    

    def obtain_detections(self, frame: np.ndarray) -> list[dict]:
        """
        Runs detection on a frame and returns the results.

        Args:
            frame: Input frame.

        Returns:
            list of detection dictionaries.
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

        Args:
            frame: Input frame.
            detections: list of detection dictionaries.

        Returns:
            list of tracked STrack objects.
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


    def process_frame(
        self,
        frame: np.ndarray,
        tracking: bool = True
    ) -> tuple[list[dict], list[STrack]]:
        """
        Runs detection (and optionally tracking) on a frame.

        Args:
            frame: Input frame.
            tracking: Whether to track objects.

        Returns:
            tuple (detections, tracks).
        """
        h, w = frame.shape[:2]

        detections: list[dict] = []

        if self.onnx:
            detections = self.model.pipeline(frame, self.conf_thresh, self.iou_thresh)

            tracking_input = np.array(
                [[*det["bbox"], det["conf"]] for det in detections],
                dtype=np.float32
            ) if detections else np.empty((0, 5), dtype=np.float32)

        else:
            results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh,
                                 max_det=self.max_det, verbose=False)[0]

            det_list = []
            track_list = []

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf >= self.conf_thresh and cls == 0:
                    det_list.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "class": cls
                    })
                    track_list.append([x1, y1, x2, y2, conf])

            detections = det_list
            tracking_input = (
                np.array(track_list, dtype=np.float32)
                if track_list else np.empty((0, 5), dtype=np.float32)
            )

        tracks: list[STrack] = []
        if tracking and tracking_input.size > 0:
            tracks = self.tracker.update(tracking_input, [h, w], [h, w])

        return detections, tracks


    def draw_detections(self, frame: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
        """
        Draws detection boxes on the frame.

        Args:
            frame (np.ndarray): Frame to annotate.
            detections (list[Dict[str, object]]): list of detection dictionaries. Each must have:
                - 'bbox': tuple of coordinates (x1, y1, x2, y2) or (x, y, w, h)
                - 'conf': Confidence score (float)
                - 'class': Class ID (int)
            onnx (bool): If True, interprets bbox as (x, y, w, h). Otherwise, (x1, y1, x2, y2).

        Returns:
            np.ndarray: Annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            conf: float = float(det["conf"])
            cls: int = int(det["class"])
            color: tuple = (0, 255, 0) if cls == 0 else (0, 0, 255)

            label: str = (
                f'{self.classes[cls]}: {conf:.2f}'
                if hasattr(self, 'classes') and cls < len(self.classes)
                else f'Class {cls} {conf:.2f}'
            )

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            cv2.rectangle(frame, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return frame


    def compute_target_box(self, detections: list[dict], p: float) -> Optional[tuple[float, float, float, float]]:
        """
        Computes and formats the smallest bounding box containing p% of detections.

        Args:
            detections: list of detection dictionaries.
            p: Fraction of boxes to keep (0-1).
            frame_shape: Shape of the frame (height, width, channels).

        Returns:
            tuple (cx, cy, w, h) of the focus box, or None if no detections.
        """
        if not detections:
            return None

        boxes = [d["bbox"] for d in detections]
        target_num = max(1, int(len(boxes) * p))
        indices = list(range(len(boxes)))

        while len(indices) > target_num:
            best_area = None
            best_idx = None
            for idx in indices:
                test = [i for i in indices if i != idx]
                xs = [boxes[i][0] for i in test] + [boxes[i][2] for i in test]
                ys = [boxes[i][1] for i in test] + [boxes[i][3] for i in test]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if best_area is None or area < best_area:
                    best_area = area
                    best_idx = idx
            indices.remove(best_idx)

        xs = [boxes[i][0] for i in indices] + [boxes[i][2] for i in indices]
        ys = [boxes[i][1] for i in indices] + [boxes[i][3] for i in indices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y

        cx, cy, w, h = enforce_aspect_ratio(cx, cy, w, h, self.target_ratio)

        return cx, cy, w, h


    def update_current_box(self, target: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return self.box_controller.update(target)


    def get_boxes(
        self,
        box_info: tuple[float, float, float, float],
        n: float,
        frame_shape: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        """
        Computes tight and expanded bounding boxes around the focus box,
        and ensures that the expanded box keeps the target aspect ratio exactly,
        always fits within the frame, and is only shifted — never clamped or distorted.
        """
        cx, cy, w, h = box_info
        frame_h, frame_w = frame_shape[:2]

        # Tight box
        tight_x1 = int(cx - w / 2)
        tight_y1 = int(cy - h / 2)
        tight_x2 = int(cx + w / 2)
        tight_y2 = int(cy + h / 2)

        # Expand
        exp_w = w * n
        exp_h = h * n

        # Adjust to target aspect ratio
        target_ratio = self.target_ratio
        if exp_w / exp_h > target_ratio:
            exp_w = exp_h * target_ratio
        else:
            exp_h = exp_w / target_ratio

        # Ensure it fits in frame — scale down if needed
        max_w = min(frame_w, frame_h * target_ratio)
        max_h = min(frame_h, frame_w / target_ratio)

        if exp_w > max_w or exp_h > max_h:
            scale = min(max_w / exp_w, max_h / exp_h)
            exp_w *= scale
            exp_h *= scale

        # Position the box centered at (cx, cy)
        exp_x1 = int(round(cx - exp_w / 2))
        exp_y1 = int(round(cy - exp_h / 2))
        exp_x2 = int(round(cx + exp_w / 2))
        exp_y2 = int(round(cy + exp_h / 2))

        # Shift if needed to keep inside frame
        shift_x = 0
        shift_y = 0
        if exp_x1 < 0:
            shift_x = -exp_x1
        elif exp_x2 > frame_w:
            shift_x = frame_w - exp_x2
        if exp_y1 < 0:
            shift_y = -exp_y1
        elif exp_y2 > frame_h:
            shift_y = frame_h - exp_y2

        exp_x1 += shift_x
        exp_x2 += shift_x
        exp_y1 += shift_y
        exp_y2 += shift_y

        return (
            (tight_x1, tight_y1, tight_x2, tight_y2),
            (exp_x1, exp_y1, exp_x2, exp_y2)
        )

    

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

        Args:
            frame: The input frame.
            max_patch_aspect_ratio: If frame width/height exceeds this, patching is triggered.
            overlap_ratio: Overlap between patches (0 to 1).
            max_patches: Maximum number of patches allowed (defaults to 2).

        Returns:
            A merged list of deduplicated detections.
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

        if filter:
            print("mask shape:", self.field_mask.shape)
            detections = filter_detections_by_mask(detections, self.field_mask)

        return detections


if __name__ == "__main__":
    # input_video = "videos/soccertrack/wide_view/videos/F_20200220_1_0030_0060.mp4"
    input_video = "videos/hqsport-clip.mp4"
    output_dir = "videos/output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # model_path: str = "models/yolov8n.pt"
    model_path: str = "models/yolo8n_static_quantized.onnx"
    frame_interval: int = 3

    onnx: bool = ".onnx" in model_path

    tracker = PlayerTracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        frame_interval=frame_interval,
        onnx=onnx,
        chosen_resolution="HD",
        input_video=input_video
    )

    output_path = os.path.join(output_dir, "output_tracking.mp4")
    tracker.process_video(
        input_video,
        output_path,
        tracking=False,
        display=True,
        real_output=False
    )
