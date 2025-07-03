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

        self.conf_thresh: float = 0.1
        self.iou_thresh: float = 0.1
        self.max_det: int = 300
        self.model: YOLO = YOLO(model_path).to(device)

        self.skip_frames: int = 3

        # Boxes
        self.n: float = 1.5 # Scale factor for expanded box
        self.p: float = 0.7 # Percetage of boxes to consider for focus box

        # Tracker configuration
        tracker_args: dict = {
            "track_thresh": self.conf_thresh,
            "track_buffer": 30,
            "match_thresh": 0.8,
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
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_id: int = 0
        tracks: list = []
        detections: list = []

        # For moving average FPS per frame (instantaneous)
        fps_window = deque(maxlen=30)

        # For overall effective FPS
        total_start_time = time.time()
        total_frames: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            if frame_id % self.skip_frames == 0:
                detections, tracks = self.process_frame(frame, tracking=tracking)

            if tracking:
                frame = self.draw_tracks(frame, tracks)
                focus_input: list[dict] = self.tracks_to_detections(tracks)
            else:
                frame = self.draw_detections(frame, detections)
                focus_input = detections
            
            frame = self.draw_boxes(focus_input, frame)

            frame, avg_fps, effective_fps, total_frames = self.compute_fps(frame, fps_window, total_start_time,
                                                                           total_frames, frame_start)

            out.write(frame)

            if display:
                cv2.imshow("Tracking" if tracking else "Detection", frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_id += 1
            if frame_id % 10 == 0:
                print(
                    f"Processed frame {frame_id} | "
                    f"FPS: {avg_fps:.2f} | "
                    f"Overall FPS: {effective_fps:.2f}"
                )

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        total_elapsed = time.time() - total_start_time
        final_fps = total_frames / total_elapsed
        print(f"Finished processing {total_frames} frames in {total_elapsed:.2f}s | Effective FPS: {final_fps:.2f}")
    

    def compute_fps(self, frame: np.ndarray, fps_window: deque,
                    total_start_time: float, total_frames: int, frame_start) -> np.ndarray:
        """
        Computes and overlays FPS on the frame.
        """
        frame_end = time.time()
        elapsed = frame_end - frame_start
        fps_render = 1 / elapsed if elapsed > 0 else 0
        fps_window.append(fps_render)
        avg_fps = sum(fps_window) / len(fps_window)

        total_frames += 1
        effective_fps = total_frames / (time.time() - total_start_time)

        # Overlay instantaneous FPS and effective FPS separately
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Overall FPS: {effective_fps:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )
        return frame, avg_fps, effective_fps, total_frames


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
        Draws detection boxes on the frame.
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
    

    def compute_focus_box(
        self,
        detections: list[dict],
        frame_shape: tuple
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        """
        Computes the smallest bounding box containing p% of detected boxes,
        and returns both the tight box and a larger concentric box scaled by factor n.

        Returns:
            (tight_box, expanded_box), each as (x1, y1, x2, y2)
        """
        if not detections:
            return None, None

        # Sort detections by confidence descending
        sorted_detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
        num_to_keep = max(1, int(len(sorted_detections) * self.p))
        selected = sorted_detections[:num_to_keep]

        # Compute tight bounding box
        x1s = [d["bbox"][0] for d in selected]
        y1s = [d["bbox"][1] for d in selected]
        x2s = [d["bbox"][2] for d in selected]
        y2s = [d["bbox"][3] for d in selected]

        tight_x1 = int(min(x1s))
        tight_y1 = int(min(y1s))
        tight_x2 = int(max(x2s))
        tight_y2 = int(max(y2s))

        # Compute center and size
        cx = (tight_x1 + tight_x2) / 2
        cy = (tight_y1 + tight_y2) / 2
        w = (tight_x2 - tight_x1) * self.n
        h = (tight_y2 - tight_y1) * self.n

        # Expanded box
        exp_x1 = int(cx - w / 2)
        exp_y1 = int(cy - h / 2)
        exp_x2 = int(cx + w / 2)
        exp_y2 = int(cy + h / 2)

        # Clip to frame boundaries
        height, width = frame_shape[:2]
        exp_x1 = max(0, exp_x1)
        exp_y1 = max(0, exp_y1)
        exp_x2 = min(width - 1, exp_x2)
        exp_y2 = min(height - 1, exp_y2)

        return (tight_x1, tight_y1, tight_x2, tight_y2), (exp_x1, exp_y1, exp_x2, exp_y2)
    

    def tracks_to_detections(self, tracks: list[STrack]) -> list[dict]:
        """
        Converts a list of STrack objects to detection-like dicts.
        """
        detections = []
        for t in tracks:
            x1, y1, w, h = t.tlwh
            x2 = x1 + w
            y2 = y1 + h
            conf = t.score if hasattr(t, "score") else 1.0
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class": 0  # Optionally track class if you want
            })
        return detections
    

    def draw_boxes(self, focus_input: list[dict], frame: np.ndarray) -> np.ndarray:
        tight_box, expanded_box = self.compute_focus_box(focus_input, frame_shape=frame.shape)

        if tight_box and expanded_box:
            # Tight box in blue
            tx1, ty1, tx2, ty2 = tight_box
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)

            # Expanded box in red
            ex1, ey1, ex2, ey2 = expanded_box
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)
        
        return frame




if __name__ == "__main__":
    input_video = "videos/soccertrack/wide_view/videos/F_20200220_1_0030_0060.mp4"
    output_dir = "videos/output_videos"
    os.makedirs(output_dir, exist_ok=True)

    model_path = "models/yolo11s_15_736_300/best.pt"

    tracker = PlayerTracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    output_path = os.path.join(output_dir, "output_tracking.mp4")
    tracker.process_video(
        input_video,
        output_path,
        tracking=True,
        display=True
    )
