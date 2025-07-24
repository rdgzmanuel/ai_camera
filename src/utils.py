import numpy as np
import time
import cv2
import os
import json
from collections import deque
from yolox.tracker.byte_tracker import STrack



def enforce_aspect_ratio(cx: float, cy: float, w: float, h: float, target_ratio: float) -> tuple[float, float, float, float]:
        """
        Adjusts width or height so that the box matches the target aspect ratio,
        centered around (cx, cy), by expanding the shorter side.

        Args:
            cx: Center x
            cy: Center y
            w: Current width
            h: Current height
            target_ratio: Desired aspect ratio (width / height)

        Returns:
            (cx, cy, adjusted_w, adjusted_h)
        """
        current_ratio = w / h if h != 0 else 0

        if current_ratio > target_ratio:
            # Too wide → increase height
            h = w / target_ratio
        else:
            # Too tall → increase width
            w = h * target_ratio

        return cx, cy, w, h


def compute_fps(
        frame: np.ndarray,
        fps_window: deque,
        total_start_time: float,
        total_frames: int,
        frame_start: float
    ) -> tuple[np.ndarray, float, float, int]:
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

        total_frames += 1
        effective_fps = total_frames / (time.time() - total_start_time)

        cv2.putText(frame, f"Overall FPS: {effective_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame, effective_fps, total_frames


def draw_tracks(frame: np.ndarray, tracks: list[STrack]) -> np.ndarray:
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
            color = get_track_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame


def get_track_color(track_id: int) -> tuple[int, int, int]:
        """
        Generates a consistent color for a track ID.

        Args:
            track_id: Unique track identifier.

        Returns:
            RGB color tuple.
        """
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


def tracks_to_detections(tracks: list[STrack]) -> list[dict]:
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


def draw_boxes(
        frame: np.ndarray,
        tight_box: tuple[int, int, int, int],
        expanded_box: tuple[int, int, int, int]
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


def non_max_merge(detections: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """
    Merge overlapping detections using a greedy IoU-based NMS.
    Assumes all detections are of the same class.

    Args:
        detections: List of detection dictionaries.
        iou_thresh: IOU threshold for suppression.

    Returns:
        Filtered list of detections.
    """
    if not detections:
        return []

    boxes: np.ndarray = np.array([d["bbox"] for d in detections])
    scores: np.ndarray = np.array([d["conf"] for d in detections])
    indices: list[int] = sorted(range(len(boxes)), key=lambda i: -scores[i])

    keep: list[dict] = []
    while indices:
        current: int = indices.pop(0)
        keep.append(detections[current])

        remaining: list[int] = []
        for idx in indices:
            if iou(boxes[current], boxes[idx]) < iou_thresh:
                remaining.append(idx)
        indices = remaining

    return keep


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """
    Computes IoU between two boxes.
    """
    x_a: float = max(box_a[0], box_b[0])
    y_a: float = max(box_a[1], box_b[1])
    x_b: float = min(box_a[2], box_b[2])
    y_b: float = min(box_a[3], box_b[3])

    inter_area: float = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area: float = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area: float = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)


def load_field_mask(input_path: str, resolution: tuple) -> np.ndarray:
    """
    Loads the field polygon from JSON and creates a binary mask.
    
    Args:
        input_path: Path to the input video.
        resolution: Tuple (width, height) of the frames.
    
    Returns:
        Binary mask with the same resolution as the video frames.
    """
    json_path = os.path.splitext(input_path)[0] + ".json"
    with open(json_path, "r") as f:
        data = json.load(f)
    polygon = [(point["x"], point["y"]) for point in data["field_coordinates"]]

    mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask


def filter_detections_by_mask(detections: list, mask: np.ndarray) -> list:
    """
    Filters detections to only those inside the field mask.
    
    Args:
        detections: List of detection dicts, each with a 'bbox' key.
        mask: Binary mask where field area is 1.
    
    Returns:
        List of filtered detections.
    """
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0] and mask[cy, cx]:
            filtered.append(det)
    return filtered
