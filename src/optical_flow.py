import cv2
import numpy as np
from typing import Optional


class OpticalFlow:
    """
    Lightweight optical flow implementation optimized for performance
    """
    
    def __init__(self):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.detection_centers = []
        self.prev_gray = None
        self.prev_detections = []
        
        self.flow_scale = 0.5
        
    def extract_detection_centers(self, detections: list[dict]) -> list[tuple[float, float]]:
        """
        Extract only center points from detections - much faster than 5 points per detection
        """
        centers = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return centers


    def downsample_frame_for_flow(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Downsample frame for flow calculation to reduce computation
        """
        h, w = gray_frame.shape
        new_h, new_w = int(h * self.flow_scale), int(w * self.flow_scale)
        return cv2.resize(gray_frame, (new_w, new_h))


    def estimate_motion_complexity(self, gray_frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> float:
        if prev_gray is None:
            return 0.0

        # Use sparse grid
        h, w = gray_frame.shape
        grid_points = np.array([
            [x, y] for y in range(40, h-40, 80)
            for x in range(40, w-40, 80)
        ], dtype=np.float32).reshape(-1, 1, 2)

        if len(grid_points) < 4:
            return 0.0

        small_gray = self.downsample_frame_for_flow(gray_frame)
        small_prev = self.downsample_frame_for_flow(prev_gray)

        if small_gray.shape != small_prev.shape:
            return 0.0  # Prevents OpenCV crash

        small_grid = grid_points * self.flow_scale

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            small_prev, small_gray, small_grid, None, **self.lk_params
        )

        valid_flow = new_points[status.flatten() == 1] - small_grid[status.flatten() == 1]
        if len(valid_flow) == 0:
            return 0.0

        motion_magnitude = np.mean(np.linalg.norm(valid_flow, axis=1)) / self.flow_scale
        return motion_magnitude


    def propagate_detections(self, gray_frame: np.ndarray) -> list[dict]:
        """
        Simplified detection propagation using only center points
        """
        if not self.prev_detections or self.prev_gray is None or not self.detection_centers:
            return []
        
        # Downsample frames for flow calculation
        small_gray = self.downsample_frame_for_flow(gray_frame)
        small_prev = self.downsample_frame_for_flow(self.prev_gray)
        
        # Scale detection centers to match downsampled frame
        small_centers = np.array(self.detection_centers, dtype=np.float32) * self.flow_scale
        small_centers = small_centers.reshape(-1, 1, 2)
        
        # Calculate optical flow
        new_centers, status, error = cv2.calcOpticalFlowPyrLK(
            small_prev, small_gray, small_centers, None, **self.lk_params
        )
        
        propagated_detections = []
        
        for i, detection in enumerate(self.prev_detections):
            if status[i][0] == 1 and error[i][0] < 50:
                # Calculate displacement in original resolution
                old_center = np.array(self.detection_centers[i])
                new_center = new_centers[i][0] / self.flow_scale
                
                displacement = new_center - old_center
                
                x1, y1, x2, y2 = detection["bbox"]
                new_x1 = x1 + displacement[0]
                new_y1 = y1 + displacement[1]
                new_x2 = x2 + displacement[0]
                new_y2 = y2 + displacement[1]
                
                frames_since_detection = getattr(self, 'frames_since_full_detection', 1)
                confidence_decay = 0.9 ** frames_since_detection
                new_conf = detection["conf"] * confidence_decay
                
                if new_conf >= 0.2:
                    propagated_detections.append({
                        "bbox": (new_x1, new_y1, new_x2, new_y2),
                        "conf": new_conf,
                        "class": detection["class"],
                        "propagated": True
                    })
        
        return propagated_detections