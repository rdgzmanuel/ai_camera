import cv2
import numpy as np
import time


class AdaptiveResolutionDetector:
    """
    Adaptively scales input resolution for YOLO based on scene complexity and performance requirements.
    """
    
    def __init__(self, base_model, min_size: int = 416, max_size: int = 832):
        self.model = base_model
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = 640  # Start with standard YOLO size
        
        # Performance tracking
        self.inference_times = []
        self.detection_counts = []
        self.target_fps = 30
        self.target_inference_time = 1.0 / self.target_fps * 0.7  # 70% of frame time for inference
        
        # Quality metrics
        self.min_detection_count = 3  # Minimum detections expected in active scenes


    def estimate_scene_complexity(self, frame: np.ndarray) -> float:
        """
        Estimate scene complexity based on edge density and motion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # Texture complexity using standard deviation
        texture_complexity = np.std(gray) / 255.0
        
        # Combine metrics
        complexity = (edge_density * 0.6 + texture_complexity * 0.4) * 100
        return complexity


    def adjust_resolution(self, inference_time: float, detection_count: int, 
                         scene_complexity: float) -> int:
        """
        Dynamically adjust input resolution based on performance and quality metrics.
        """
        self.inference_times.append(inference_time)
        self.detection_counts.append(detection_count)
        
        # Keep only recent history
        if len(self.inference_times) > 10:
            self.inference_times.pop(0)
            self.detection_counts.pop(0)
        
        avg_inference_time = np.mean(self.inference_times)
        avg_detection_count = np.mean(self.detection_counts)
        
        # Decision logic
        new_size = self.current_size
        
        # If inference is too slow, reduce resolution
        if avg_inference_time > self.target_inference_time * 1.2:
            new_size = max(self.min_size, self.current_size - 64)

        # If we have spare computational capacity and low detection count or high complexity
        elif (avg_inference_time < self.target_inference_time * 0.8 and 
              (avg_detection_count < self.min_detection_count or scene_complexity > 30)):
            new_size = min(self.max_size, self.current_size + 64)

        # Smooth transitions
        if abs(new_size - self.current_size) > 0:
            self.current_size = new_size
            print(f"Adjusted resolution to {self.current_size}x{self.current_size}")

        return self.current_size


    def detect_with_adaptive_resolution(self, frame: np.ndarray, 
                                      conf_thresh: float = 0.3,
                                      iou_thresh: float = 0.1) -> tuple[list[dict], float]:
        """
        Run detection with adaptive resolution scaling.
        """
        # Estimate scene complexity
        scene_complexity = self.estimate_scene_complexity(frame)
        
        # Resize frame for inference
        h, w = frame.shape[:2]
        scale_factor = self.current_size / max(h, w)
        
        if scale_factor < 1.0:
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized_frame = cv2.resize(frame, (new_w, new_h))
        else:
            resized_frame = frame
            scale_factor = 1.0
        
        # Run inference
        start_time = time.time()
        
        if hasattr(self.model, 'pipeline'):  # ONNX model
            detections = self.model.pipeline(resized_frame, conf_thresh, iou_thresh)
        else:  # PyTorch YOLO
            results = self.model(resized_frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf >= conf_thresh and cls == 0:
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "class": cls
                    })
        
        inference_time = time.time() - start_time
        
        # Scale detections back to original frame size
        if scale_factor < 1.0:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (x1/scale_factor, y1/scale_factor, x2/scale_factor, y2/scale_factor)
        
        # Adjust resolution for next frame
        self.adjust_resolution(inference_time, len(detections), scene_complexity)
        
        return detections, inference_time
