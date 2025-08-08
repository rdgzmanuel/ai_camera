import onnxruntime
import numpy as np
import cv2


class OnnxDetector:
    def __init__(self, model_path: str):
        # Use optimized ONNX Runtime session
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4  # Tune based on CPU cores

        self.session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def preprocessor(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        image_data = resized.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))[np.newaxis, ...]
        return image_data

    def detector(self, image_data: np.ndarray) -> list:
        return self.session.run(self.output_names, {"images": image_data})

    def postprocessor(self, results, frame: np.ndarray, confidence: float, iou: float) -> list[dict]:
        img_height, img_width = frame.shape[:2]
        outputs = np.transpose(np.squeeze(results[0]))  # shape: (N, 85)
        rows = outputs.shape[0]

        boxes = []
        scores = []
        detection_results = []

        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        for i in range(rows):
            person_conf = outputs[i][4]  # confidence for class 0 (person)

            if person_conf >= confidence:
                x, y, w, h = outputs[i][:4]

                x1: int = int((x - w / 2) * x_factor)
                y1: int = int((y - h / 2) * y_factor)
                width: int = int(w * x_factor)
                height: int = int(h * y_factor)

                x2: int = x1 + width
                y2: int = y1 + height

                boxes.append([x1, y1, x2, y2])  # For NMS
                scores.append(float(person_conf))

                detection_results.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": person_conf,
                    "class": 0  # Optional: restore if downstream code needs it
                })

        indices = fast_nms(boxes, scores, confidence, iou)

        final_results = [detection_results[i] for i in indices]
        return final_results


    def pipeline(self, frame: np.ndarray, confidence_thrs: float, iou_thrs: float) -> list[dict]:
        image_data = self.preprocessor(frame)
        results = self.detector(image_data)
        return self.postprocessor(results, frame, confidence_thrs, iou_thrs)


def fast_nms(boxes, scores, conf_threshold, iou_threshold):
    boxes = np.array(boxes)
    scores = np.array(scores)

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
