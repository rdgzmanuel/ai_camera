import os
import time
from collections import deque
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker, STrack
from ultralytics import YOLO


class PlayerTracker:
    """
    A class for detecting and optionally tracking objects using YOLOv8/YOLOv11 and BYTETrack.
    """

    def __init__(self, model_path: str = "models/yolo11s.pt", device: str = "cpu") -> None:
        """
        Initializes the PlayerTracker with YOLO model and (optionally) BYTETrack.
        """
        self.device: str = device

        self.conf_thresh: float = 0.2
        self.iou_thresh: float = 0.1
        self.max_det: int = 300
        self.model: YOLO = YOLO(model_path).to(device)

        self.skip_frames: int = 5

        # Tracker configuration
        tracker_args: dict = {
            "track_thresh": self.conf_thresh,
            "track_buffer": 30,
            "match_thresh": 1.0,
            "min_box_area": 5,
            "mot20": False,
        }
        self.tracker: BYTETracker = BYTETracker(SimpleNamespace(**tracker_args), frame_rate=30)


    def process_video(
        self,
        input_path: str,
        output_path: str,
        tracking: bool = True,
        display: bool = False
    ) -> None:
        """
        Processes video with detection and optional tracking, and prints average FPS.
        """
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_id = 0
        tracks = []
        detections = []

        # Keep last N frame times for a moving average
        fps_window = deque(maxlen=30)

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % self.skip_frames == 0:
                detections, tracks = self.process_frame(frame, tracking=tracking)

            if tracking:
                frame = self.draw_tracks(frame, tracks)
            else:
                frame = self.draw_detections(frame, detections)

            frame_end = time.time()
            elapsed = frame_end - frame_start
            fps_render = 1 / elapsed if elapsed > 0 else 0
            fps_window.append(fps_render)
            avg_fps = sum(fps_window) / len(fps_window)

            # Overlay average FPS
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            out.write(frame)

            if display:
                cv2.imshow("Tracking" if tracking else "Detection", frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_id += 1
            if frame_id % 10 == 0:
                print(f"Processed frame {frame_id} | Avg FPS: {avg_fps:.2f}")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()



    def process_frame(
        self,
        frame: np.ndarray,
        tracking: bool = True
    ) -> tuple[list[dict], list[STrack]]:
        """
        Runs detection (and optionally tracking) on a frame.

        Args:
            frame (np.ndarray): Input frame.
            tracking (bool): Whether to track.

        Returns:
            detections (list[dict]): Raw detections.
            tracks (list[STrack]): Active tracks (if tracking), else empty.
        """
        original_height, original_width = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=self.max_det,
            verbose=False
        )[0]

        detections = []
        detections_for_tracking = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf >= self.conf_thresh:
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "class": cls
                })
                detections_for_tracking.append([x1, y1, x2, y2, conf])

        if tracking and detections_for_tracking:
            det_tensor = torch.tensor(detections_for_tracking)
            tracks = self.tracker.update(
                det_tensor,
                [original_height, original_width],
                [original_height, original_width]
            )
        else:
            tracks = []

        return detections, tracks

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draws detections on the frame.
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

    def draw_tracks(self, frame: np.ndarray, tracks: list[STrack]) -> np.ndarray:
        """
        Draws tracks on the frame.
        """
        for track in tracks:
            x1, y1, w, h = map(int, track.tlwh)
            track_id = track.track_id
            color = self.get_track_color(track_id)

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def get_track_color(self, track_id: int) -> tuple[int, int, int]:
        """
        Generates a consistent color for a track ID.
        """
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


if __name__ == "__main__":
    input_video = "videos/soccertrack/wide_view/videos/F_20200220_1_0030_0060.mp4"
    output_dir = "videos/output_videos"
    os.makedirs(output_dir, exist_ok=True)

    model_path = "models/yolo11s_14_1024_200/best.pt"

    tracker = PlayerTracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    output_path = os.path.join(output_dir, "output_tracking.mp4")
    tracker.process_video(
        input_video,
        output_path,
        tracking=False,
        display=True
    )
