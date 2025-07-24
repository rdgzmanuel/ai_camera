import cv2
import numpy as np
import os
import json


class SoccerFieldDetector:
    def __init__(self, video_path: str):
        """
        Initialize the SoccerFieldDetector with the path to the video.
        
        Parameters:
            video_path (str): Path to the input video file.
        """
        self.video_path: str = video_path


    def extract_representative_frame(self) -> np.ndarray:
        """
        Extracts the first clear frame from the video.
        
        Returns:
            frame (np.ndarray): A single frame from the video.
        """
        cap = cv2.VideoCapture(self.video_path)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise RuntimeError("Failed to read frame from video.")
        return frame


    def mask_green_field(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies green color masking to extract the field area with shadow robustness.
        
        Parameters:
            frame (np.ndarray): Input image (BGR).
        
        Returns:
            mask (np.ndarray): Binary mask of green areas.
        """
        # Optional contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Wider green bounds
        lower_green = np.array([25, 10, 10])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        mask = cv2.dilate(mask, kernel, iterations=8)           # Expand field
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("field_coordinates/green_mask_debug.png", mask)

        return mask


    def detect_field_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Finds the largest external contour in the green mask as the soccer field.
        
        Parameters:
            mask (np.ndarray): Binary mask of green field.
        
        Returns:
            approx (np.ndarray): Approximated polygon contour of the field.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found.")

        height, width = mask.shape
        min_area = 0.25 * width * height  # Must occupy at least 25% of the frame

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < min_area:
            raise RuntimeError("No sufficiently large field contour found.")

        # Optional: simplify to polygon
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        return approx


    def extract_field_coordinates(self, polygon: np.ndarray) -> list:
        """
        Extracts and formats polygon points as coordinate tuples.
        
        Parameters:
            polygon (np.ndarray): Approximated polygon contour.
        
        Returns:
            coordinates (list): List of (x, y) tuples.
        """
        return [(int(point[0][0]), int(point[0][1])) for point in polygon]


    def visualize_polygon(self, frame: np.ndarray, polygon: list) -> np.ndarray:
        """
        Draws the detected polygon on the image.
        
        Parameters:
            frame (np.ndarray): Original image.
            polygon (list): List of (x, y) tuples of field boundary.
        
        Returns:
            image (np.ndarray): Image with polygon overlay.
        """
        vis = frame.copy()
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
        return vis
    

    def save_coordinates_to_json(self, coordinates: list) -> None:
        """
        Saves the polygon coordinates to a JSON file with the same name as the video.

        Parameters:
            coordinates (list): List of (x, y) tuples outlining the field.
        """
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        json_path = f"field_coordinates/{base_name}.json"

        # Convert list of tuples to list of dicts for better JSON compatibility
        coord_dicts = [{"x": x, "y": y} for (x, y) in coordinates]

        with open(json_path, "w") as f:
            json.dump({"field_coordinates": coord_dicts}, f, indent=4)

        print(f"Saved coordinates to {json_path}")


    def run(self) -> list:
        """
        Executes the full pipeline to detect the soccer field.

        Returns:
            field_coordinates (list): List of (x, y) coordinates outlining the field.
        """
        frame = self.extract_representative_frame()
        green_mask = self.mask_green_field(frame)
        polygon = self.detect_field_contour(green_mask)
        coordinates = self.extract_field_coordinates(polygon)

        # Save visualization
        visual = self.visualize_polygon(frame, coordinates)
        cv2.imwrite("field_coordinates/detected_field.png", visual)

        # Save to JSON
        self.save_coordinates_to_json(coordinates)

        return coordinates


if __name__ == "__main__":
    input_video: str = "videos/hqsport-clip.mp4"
    detector = SoccerFieldDetector(input_video)
    coords = detector.run()
    print("Detected field coordinates:", coords)
