import os
import time
from collections import deque
from typing import Optional, List, Tuple
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker, STrack
from ultralytics import YOLO


class PlayerTracker:
    """
    A class for detecting and optionally tracking moving objects in video streams.
    Computes dynamic focus boxes and smooth camera movement.
    """

    def __init__(self, model_path: str = "models/yolo11n_nb_1280_300/best.pt",
                 device: str = "cpu",
                 frame_interval: int = 1,
                 chosen_resolution: str = "FHD") -> None:
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
        self.model: YOLO = YOLO(model_path).to(device)

        self.frame_interval: int = frame_interval
        self.n: float = 1.75  # Scale factor for expanded box
        self.p: float = 0.85  # Percentage of boxes to consider
        self.default_fps: int = 25
        self.frame_rate: int = self.default_fps // self.frame_interval

        tracker_args: dict = {
            "track_thresh": self.conf_thresh,
            "track_buffer": 30,
            "match_thresh": 0.95,
            "min_box_area": 5,
            "mot20": False,
        }
        self.tracker: BYTETracker = BYTETracker(SimpleNamespace(**tracker_args), frame_rate=self.frame_rate)

        # Visualization parameters
        self.aspect_ratio: Tuple[int, int] = (16, 9)
        self.target_ratio: float = self.aspect_ratio[0] / self.aspect_ratio[1]
        self.current_box_info: Optional[Tuple[float, float, float, float]] = None
        self.max_speed: float = 7.0

        self.resolutions: dict[str, tuple[int, int]] = {
            "HD": (1280, 720),
            "FHD": (1920, 1080),
            "QHD": (2560, 1440),
            "4K": (3840, 2160),
        }
        self.resolution: tuple[int, int] = self.resolutions[chosen_resolution]


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
        box_info: Optional[Tuple[float, float, float, float]] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start: float = time.time()

            if frame_id % self.frame_interval == 0:
                detections, tracks = self.process_frame(frame, tracking=tracking)
                if tracking:
                    focus_input = self.tracks_to_detections(tracks)
                else:
                    focus_input = detections

                box_info = self.compute_focus_box(
                    focus_input,
                    self.p
                )

            if box_info is not None:
                updated_box_info = self.update_box_info(box_info)
                tight_box, expanded_box = self.get_boxes(updated_box_info, self.n, frame.shape)

                if real_output:
                    # Crop to expanded box only
                    ex1, ey1, ex2, ey2 = expanded_box
                    frame = frame[ey1:ey2, ex1:ex2]

                    # Optionally, resize to original resolution so all outputs are same size
                    frame = cv2.resize(frame, self.resolution)
                else:
                    if tracking:
                        frame = self.draw_tracks(frame, tracks)
                    else:
                        frame = self.draw_detections(frame, detections)

                    frame = self.draw_boxes(frame, tight_box, expanded_box)

            frame, avg_fps, effective_fps, total_frames = self.compute_fps(
                frame, fps_window, total_start_time, total_frames, frame_start
            )

            out.write(frame)

            if display:
                cv2.imshow("Tracking" if tracking else "Detection", frame)
                if cv2.waitKey(1) == 27:
                    break

            if frame_id % 10 == 0:
                print(
                    f"Frame {frame_id} | FPS: {avg_fps:.2f} | Overall FPS: {effective_fps:.2f}"
                )
            frame_id += 1

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        total_elapsed = time.time() - total_start_time
        final_fps = total_frames / total_elapsed
        print(f"Processed {total_frames} frames in {total_elapsed:.2f}s | Effective FPS: {final_fps:.2f}")


    def compute_fps(
        self,
        frame: np.ndarray,
        fps_window: deque,
        total_start_time: float,
        total_frames: int,
        frame_start: float
    ) -> Tuple[np.ndarray, float, float, int]:
        """
        Computes and overlays FPS metrics.

        Args:
            frame: Frame to annotate.
            fps_window: Sliding window of FPS.
            total_start_time: Start timestamp.
            total_frames: Number of frames processed.
            frame_start: Timestamp of frame start.

        Returns:
            Tuple (annotated frame, avg FPS, effective FPS, updated total_frames).
        """
        elapsed = time.time() - frame_start
        fps_render = 1.0 / elapsed if elapsed > 0 else 0
        fps_window.append(fps_render)
        avg_fps = sum(fps_window) / len(fps_window)

        total_frames += 1
        effective_fps = total_frames / (time.time() - total_start_time)

        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Overall FPS: {effective_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame, avg_fps, effective_fps, total_frames


    def process_frame(
        self,
        frame: np.ndarray,
        tracking: bool = True
    ) -> Tuple[List[dict], List[STrack]]:
        """
        Runs detection (and optionally tracking) on a frame.

        Args:
            frame: Input frame.
            tracking: Whether to track objects.

        Returns:
            Tuple (detections, tracks).
        """
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=self.max_det,
            verbose=False
        )[0]

        detections: List[dict] = []
        tracking_input: List[list] = []
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
                tracking_input.append([x1, y1, x2, y2, conf])

        tracks: List[STrack] = []
        if tracking and tracking_input:
            det_tensor = torch.tensor(tracking_input)
            tracks = self.tracker.update(det_tensor, [h, w], [h, w])

        return detections, tracks


    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draws detection boxes on the frame.

        Args:
            frame: Frame to annotate.
            detections: List of detection dictionaries.

        Returns:
            Annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["conf"]
            cls = det["class"]
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            label = f"Class {cls} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


    def draw_tracks(self, frame: np.ndarray, tracks: List[STrack]) -> np.ndarray:
        """
        Draws tracked objects on the frame.

        Args:
            frame: Frame to annotate.
            tracks: List of STrack objects.

        Returns:
            Annotated frame.
        """
        for track in tracks:
            x1, y1, w, h = map(int, track.tlwh)
            track_id = track.track_id
            color = self.get_track_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Generates a consistent color for a track ID.

        Args:
            track_id: Unique track identifier.

        Returns:
            RGB color tuple.
        """
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


    def tracks_to_detections(self, tracks: List[STrack]) -> List[dict]:
        """
        Converts tracked objects to detection-like dictionaries.

        Args:
            tracks: List of STrack objects.

        Returns:
            List of detection dictionaries.
        """
        return [{
            "bbox": (
                t.tlwh[0],
                t.tlwh[1],
                t.tlwh[0] + t.tlwh[2],
                t.tlwh[1] + t.tlwh[3]
            ),
            "conf": getattr(t, "score", 1.0),
            "class": 0
        } for t in tracks]


    def compute_focus_box(
        self,
        detections: List[dict],
        p: float
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Computes the smallest bounding box containing p% of detections.

        Args:
            detections: List of detection dictionaries.
            p: Fraction of boxes to keep (0-1).
            frame_shape: Shape of the frame (height, width, channels).

        Returns:
            Tuple (cx, cy, w, h) of the focus box, or None if no detections.
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

        if h == 0 or w / h > self.target_ratio:
            h = max(1e-5, w / self.target_ratio)
        else:
            w = max(1e-5, h * self.target_ratio)

        return cx, cy, w, h


    def update_box_info(
        self,
        box_info: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Smoothly interpolates the box parameters toward a target box.

        Args:
            box_info: Target (cx, cy, w, h).

        Returns:
            Updated (cx, cy, w, h).
        """
        if self.current_box_info is None:
            self.current_box_info = box_info
            return box_info

        updated = []
        for cur, target in zip(self.current_box_info, box_info):
            diff = target - cur
            if abs(diff) <= self.max_speed:
                new_val = target
            else:
                new_val = cur + np.sign(diff) * self.max_speed
            updated.append(new_val)
        
        cx, cy, w, h = updated
        target_ratio = self.aspect_ratio[0] / self.aspect_ratio[1]
        if h == 0 or w / h > target_ratio:
            h = max(1e-5, w / target_ratio)
        else:
            w = max(1e-5, h * target_ratio)

        updated = (cx, cy, w, h)

        self.current_box_info = updated
        return self.current_box_info


    def get_boxes(
        self,
        box_info: Tuple[float, float, float, float],
        n: float,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Computes the tight and expanded bounding boxes.

        Args:
            box_info: (cx, cy, w, h) box parameters.
            n: Scale factor for expansion.
            frame_shape: Shape of the frame.

        Returns:
            Tuple (tight_box, expanded_box), each as (x1, y1, x2, y2).
        """
        cx, cy, w, h = box_info
        height, width = frame_shape[:2]

        tight_x1 = int(cx - w / 2)
        tight_y1 = int(cy - h / 2)
        tight_x2 = int(cx + w / 2)
        tight_y2 = int(cy + h / 2)

        exp_w = w * n
        exp_h = h * n
        exp_x1 = int(cx - exp_w / 2)
        exp_y1 = int(cy - exp_h / 2)
        exp_x2 = int(cx + exp_w / 2)
        exp_y2 = int(cy + exp_h / 2)

        # Clip to frame bounds
        tight_x1 = max(0, tight_x1)
        tight_y1 = max(0, tight_y1)
        tight_x2 = min(width - 1, tight_x2)
        tight_y2 = min(height - 1, tight_y2)

        exp_x1 = max(0, exp_x1)
        exp_y1 = max(0, exp_y1)
        exp_x2 = min(width - 1, exp_x2)
        exp_y2 = min(height - 1, exp_y2)

        return (tight_x1, tight_y1, tight_x2, tight_y2), (exp_x1, exp_y1, exp_x2, exp_y2)


    def draw_boxes(
        self,
        frame: np.ndarray,
        tight_box: Tuple[int, int, int, int],
        expanded_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Draws the tight and expanded boxes on the frame.

        Args:
            frame: Frame to annotate.
            tight_box: Coordinates (x1, y1, x2, y2).
            expanded_box: Coordinates (x1, y1, x2, y2).

        Returns:
            Annotated frame.
        """
        tx1, ty1, tx2, ty2 = tight_box
        ex1, ey1, ex2, ey2 = expanded_box

        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)
        return frame


if __name__ == "__main__":
    input_video = "videos/soccertrack/wide_view/videos/F_20200220_1_0030_0060.mp4"
    input_video = "videos/hqsport-clip-3.mp4"
    output_dir = "videos/output_videos"
    os.makedirs(output_dir, exist_ok=True)

    model_path: str = "models/yolo11n.pt"
    frame_interval: int = 3

    tracker = PlayerTracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        frame_interval=frame_interval,
    )

    output_path = os.path.join(output_dir, "output_tracking.mp4")
    tracker.process_video(
        input_video,
        output_path,
        tracking=True,
        display=True,
        real_output=False
    )
