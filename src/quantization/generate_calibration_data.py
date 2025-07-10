import cv2
import os

video_path = "videos/hqsport-clip-3.mp4"
output_dir = "calibration_data"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
save_every = 15  # Save one frame every 15 frames

while True:
    ret, frame = cap.read()
    if not ret or frame_id > 200:
        break

    if frame_id % save_every == 0:
        path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(path, frame)

    frame_id += 1

cap.release()
