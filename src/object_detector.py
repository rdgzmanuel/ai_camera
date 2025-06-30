import os
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker, STrack
from ultralytics import YOLO


class PlayerTracker:
    """
    A class for detecting and tracking objects using YOLOv8 and BYTETrack.
    
    Attributes:
        device (str): Device for inference ('cuda' or 'cpu').
        model (YOLO): Ultralytics YOLOv11 model.
        tracker (BYTETracker): BYTETrack tracker instance.
    """

    def __init__(self, model_path: str = "models/yolo11s.pt", device: str = "cpu") -> None:
        """
        Initializes the PlayerTracker with a YOLOv11 model and BYTETrack tracker.

        Args:
            model_path (str): Path to the YOLOv8 model file (.pt).
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        self.device: str = device

        self.conf_thresh: float = 0.0  # Confidence threshold for detections
        self.iou_thresh: float = 0.1  # IoU threshold for NMS
        self.classes: list[int] = [0]  # COCO class for 'person'
        self.max_det: int = 300  # Candidates per frame for tracking
        self.model: YOLO = YOLO(model_path).to(device)

        tracker_args: dict = {
            "track_thresh": self.conf_thresh, # Detection confidence threshold for tracking 0.3
            "track_buffer": 30, # Buffer size for tracking 30
            "match_thresh": 0.7, # Matching threshold for tracking 0.6
            "min_box_area": 5, # Minimum box area for tracking 5
            "mot20": False,
        }
        self.tracker: BYTETracker = BYTETracker(SimpleNamespace(**tracker_args), frame_rate=30)

        self.resized_shape: tuple[int, int] = (1920, 1080)  # Resize shape for YOLOv11 input
    

    def process_video(self, input_path: str, output_path: str, display: bool = False) -> None:
        """
        Processes a video file and applies detection and tracking frame-by-frame.

        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to save the output annotated video.
            display (bool): Whether to show the video in real time.
        """
        cap = cv2.VideoCapture(input_path)
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # frame = cv2.resize(frame, self.resized_shape)

            tracks = self.process_frame(frame)
            frame = self.draw_tracks(frame, tracks)
            out.write(frame)

            if display:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_id += 1
            if frame_id % 10 == 0:
                print(f"Processed frame {frame_id}")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        return None


    def process_frame(self, frame: np.ndarray) -> list[STrack]:
        """
        Runs object detection and tracking on a single frame using an upscaled, contrast-enhanced image.
        Detection runs on enhanced & upscaled image, but tracking is mapped back to the original size.

        Args:
            frame (np.ndarray): Original input frame (BGR format).

        Returns:
            list[STrack]: List of active object tracks.
        """
        original_height, original_width = frame.shape[:2]

        # Enhance contrast
        # enhanced_frame: np.ndarray = self.enhance_contrast(frame)

        # Upscale the frame
        # scale_factor: float = 2
        # upscaled_frame = cv2.resize(enhanced_frame, None, fx=scale_factor, fy=scale_factor)

        # Run detection on upscaled frame
        results = self.model(frame,
                            conf=self.conf_thresh,
                            iou=self.iou_thresh,
                            max_det=self.max_det,
                            verbose=False)[0]

        detections: list[list[float]] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf: float = float(box.conf[0])
            cls: int = int(box.cls[0])

            # Scale boxes back to original frame size
            # x1 /= scale_factor
            # y1 /= scale_factor
            # x2 /= scale_factor
            # y2 /= scale_factor

            # if conf < 0.25:
            #     continue

            detections.append([x1, y1, x2, y2, conf, cls])

        if not detections:
            return []

        det_tensor = torch.tensor(detections)
        tracks: list[STrack] = self.tracker.update(
            det_tensor,
            [original_height, original_width],
            [original_height, original_width]
        )

        print(f"[TRACKING] Detections in: {len(detections)} → Tracks out: {len(tracks)}")
        for track in tracks:
            print(f"ID: {track.track_id}, BBox: {track.tlwh}")

        return tracks



    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances the contrast of an image using CLAHE in the LAB color space.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Contrast-enhanced BGR image.
        """
        lab: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l: np.ndarray
        a: np.ndarray
        b: np.ndarray
        l, a, b = cv2.split(lab)

        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl: np.ndarray = clahe.apply(l)

        enhanced_lab: np.ndarray = cv2.merge((cl, a, b))
        enhanced_img: np.ndarray = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_img



    def draw_tracks(self, frame: np.ndarray, tracks: list[STrack]) -> np.ndarray:
        """
        Draws bounding boxes and track IDs on a frame.

        Args:
            frame (np.ndarray): Input image frame (BGR format).
            tracks (list[STrack]): List of tracked objects.

        Returns:
            np.ndarray: Annotated image frame.
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
        Generates a consistent color for a given track ID.

        Args:
            track_id (int): Unique track ID.

        Returns:
            tuple[int, int, int]: RGB color.
        """
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


if __name__ == "__main__":
    input_video: str = "videos/soccertrack/wide_view/videos/F_20200220_1_0030_0060.mp4"
    output_dir: str = "videos/detection_videos"
    os.makedirs(output_dir, exist_ok=True)

    models_folder: str = "models"
    model_name: str = "yolo11m.pt"
    model_path: str = os.path.join(models_folder, model_name)

    # model_path: str = "models/yolov11m_3/last.pt"

    tracker = PlayerTracker(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    output_path: str = os.path.join(output_dir, "tracked_output.mp4")
    tracker.process_video(input_video, output_path, display=True)
