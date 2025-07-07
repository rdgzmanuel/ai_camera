import os
import time
from collections import deque
from typing import Optional, List, Tuple
import cv2
import numpy as np


class MotionDetector:
    """
    A class for detecting moving objects in video streams using background subtraction.
    Computes dynamic focus boxes and smooth camera movement.
    """

    def __init__(
        self,
        version: str = "MOG2",
        min_area: int = 500,
        chosen_resolution: str = "FHD"
    ) -> None:
        """
        Initializes the MotionDetector.

        Args:
            version: 'MOG' or 'MOG2' background subtractor.
            min_area: Minimum contour area to consider as detection.
            chosen_resolution: Output resolution.
        """
        if version == "MOG":
            self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        elif version == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        else:
            raise ValueError("version must be 'MOG' or 'MOG2'")

        self.min_area: int = min_area

        self.skip_frames: int = 3
        self.n: float = 1.75  # Scale factor for expanded box
        self.p: float = 0.75  # Percentage of boxes to consider

        # Visualization parameters
        self.aspect_ratio: Tuple[int, int] = (16, 9)
        self.current_box_info: Optional[Tuple[float, float, float, float]] = None
        self.max_speed: float = 1.0

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
        display: bool = False,
        real_output: bool = False
    ) -> None:
        """
        Processes a video file for motion detection and writes output video.

        Args:
            input_path: Path to the input video.
            output_path: Path to save the output video.
            display: Whether to show live display window.
            real_output: Whether to crop the output to the expanded box.
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

            if frame_id % self.skip_frames == 0:
                detections = self.process_frame(frame)
                box_info = self.compute_focus_box(
                    detections,
                    self.p)

            if box_info is not None:
                # updated_box_info = self.update_box_info(box_info)
                tight_box, expanded_box = self.get_boxes(box_info, self.n, frame.shape)

                if real_output:
                    ex1, ey1, ex2, ey2 = expanded_box
                    frame = frame[ey1:ey2, ex1:ex2]
                    frame = cv2.resize(frame, self.resolution)
                else:
                    frame = self.draw_detections(frame, detections)
                    frame = self.draw_boxes(frame, tight_box, expanded_box)

            frame, avg_fps, effective_fps, total_frames = self.compute_fps(
                frame, fps_window, total_start_time, total_frames, frame_start
            )

            out.write(frame)

            if display:
                cv2.imshow("Motion Detection", frame)
                if cv2.waitKey(1) == 27:
                    break

            if frame_id % 10 == 0:
                print(
                    f"Frame {frame_id} | FPS: {avg_fps:.2f} | Overall FPS: {effective_fps:.2f}"
                )
            frame_id += 1

            time.sleep(0.05)

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        total_elapsed = time.time() - total_start_time
        final_fps = total_frames / total_elapsed
        print(f"Processed {total_frames} frames in {total_elapsed:.2f}s | Effective FPS: {final_fps:.2f}")


    def process_frame(
        self,
        frame: np.ndarray
    ) -> List[dict]:
        """
        Applies background subtraction and extracts motion detections.

        Args:
            frame: Input frame.

        Returns:
            List of detection dictionaries, each containing:
                - 'bbox': (x1, y1, x2, y2)
                - 'conf': confidence (always 1.0)
                - 'class': class ID (always 0)
        """
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[dict] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    "bbox": (x, y, x + w, y + h),
                    "conf": 1.0,
                    "class": 0
                })

        return detections


    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draws detection bounding boxes on the frame.

        Args:
            frame: Frame to annotate.
            detections: List of detection dictionaries.

        Returns:
            Annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame


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
            fps_window: Sliding window of recent FPS values.
            total_start_time: Timestamp of the start of processing.
            total_frames: Number of processed frames.
            frame_start: Timestamp when current frame processing started.

        Returns:
            Tuple containing:
                - Annotated frame.
                - Average FPS over window.
                - Overall FPS.
                - Updated total_frames count.
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

        target_ratio = self.aspect_ratio[0] / self.aspect_ratio[1]
        if h == 0 or w / h > target_ratio:
            h = max(1e-5, w / target_ratio)
        else:
            w = max(1e-5, h * target_ratio)

        return cx, cy, w, h

    
    def update_box_info(
        self,
        box_info: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Smoothly interpolates the current box parameters toward a target box.

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
            print(diff)
            if abs(diff) <= self.max_speed:
                new_val = target
            else:
                new_val = cur + np.sign(diff) * self.max_speed
            updated.append(new_val)

        self.current_box_info = tuple(updated)
        return self.current_box_info



    def get_boxes(
        self,
        box_info: Tuple[float, float, float, float],
        n: float,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Computes tight and expanded bounding boxes around the focus area.

        Args:
            box_info: (cx, cy, w, h) of the focus box.
            n: Scale factor for expanded box.
            frame_shape: Frame dimensions.

        Returns:
            Tuple of tight box and expanded box coordinates.
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
        Draws the tight and expanded bounding boxes on the frame.

        Args:
            frame: Frame to annotate.
            tight_box: Tight bounding box coordinates.
            expanded_box: Expanded bounding box coordinates.

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
    input_video = "videos/hqsport-clip.mp4"
    output_dir = "videos/output_videos"
    os.makedirs(output_dir, exist_ok=True)

    detector = MotionDetector(
        version="MOG2",
        min_area=50
    )

    output_path = os.path.join(output_dir, "output_motion.mp4")
    detector.process_video(
        input_video,
        output_path,
        display=True,
        real_output=False
    )
