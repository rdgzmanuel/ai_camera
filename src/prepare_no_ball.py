import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import pandas as pd


def convert_to_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converts absolute bounding box coordinates to normalized YOLO format.
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h


def prepare_yolo_dataset(
    csv_dir: str,
    video_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    frame_step: int = 5
) -> None:
    """
    Converts soccer CSV annotations to YOLO format, sampling every N frames,
    discarding invalid frames, splitting into train/val sets, and creating
    a clean YOLO dataset directory structure.

    Args:
        csv_dir (str): Path to directory containing CSV annotation files.
        video_dir (str): Path to directory containing video files.
        output_dir (str): Path to output root directory for YOLO dataset.
        train_ratio (float): Proportion of images to use for training set.
        frame_step (int): Interval for frame sampling (e.g., 5 = every 5th frame).
    """
    tmp_images: Path = Path(output_dir) / "_all_images"
    tmp_labels: Path = Path(output_dir) / "_all_labels"
    tmp_images.mkdir(parents=True, exist_ok=True)
    tmp_labels.mkdir(parents=True, exist_ok=True)

    csv_dir_path = Path(csv_dir)
    video_dir_path = Path(video_dir)
    output_dir_path = Path(output_dir)

    all_saved_frames: int = 0

    for csv_file in csv_dir_path.glob("*.csv"):
        video_file = video_dir_path / (csv_file.stem + ".mp4")
        print(f"\n🔹 Processing CSV: {csv_file.name}")
        if not video_file.exists():
            print(f"⚠️  Warning: Video not found for {csv_file.name}")
            continue

        print(f"🔹 Loading video: {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"❌ Error: Failed to open video {video_file}")
            continue

        df: pd.DataFrame = pd.read_csv(csv_file, header=None)
        object_ids: List[str] = df.iloc[0].tolist()
        data: pd.DataFrame = df.iloc[2:].reset_index(drop=True)

        saved_frames: int = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Match annotation row (if exists)
            if frame_idx >= len(data):
                continue

            row = data.iloc[frame_idx]
            img_h, img_w = frame.shape[:2]
            image_name: str = f"{csv_file.stem}_frame{frame_idx:04d}.jpg"
            label_name: str = image_name.replace(".jpg", ".txt")

            boxes: List[str] = []
            has_invalid: bool = False

            for col in range(1, len(row) - 3, 4):
                obj_id = object_ids[col]
                # Ignore any BALL annotations
                if "BALL" in str(obj_id).upper():
                    continue

                try:
                    h = float(row[col])
                    x = float(row[col + 1])
                    y = float(row[col + 2])
                    w = float(row[col + 3])
                except (ValueError, IndexError):
                    has_invalid = True
                    break

                if any(pd.isna(v) for v in [x, y, w, h]) or w <= 0 or h <= 0:
                    has_invalid = True
                    break

                # All remaining objects are players (class 0)
                cls: int = 0
                x_center, y_center, w_norm, h_norm = convert_to_yolo(x, y, w, h, img_w, img_h)
                boxes.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if has_invalid or not boxes:
                continue  # skip this frame

            cv2.imwrite(str(tmp_images / image_name), frame)
            with open(tmp_labels / label_name, "w") as f:
                for line in boxes:
                    f.write(line + "\n")

            saved_frames += 1

        cap.release()
        all_saved_frames += saved_frames
        print(f"✅ {saved_frames} valid frames saved from {csv_file.name}")

    if all_saved_frames == 0:
        print("❌ No valid frames were saved from any CSV. Exiting.")
        return

    print(f"\n🔹 Total valid frames: {all_saved_frames}")
    print("🔹 Splitting dataset into train and val...")

    images: List[str] = [f.name for f in tmp_images.glob("*.jpg")]
    random.shuffle(images)
    train_size: int = int(len(images) * train_ratio)
    train_images: List[str] = images[:train_size]
    val_images: List[str] = images[train_size:]

    print(f"✅ Train images: {len(train_images)}")
    print(f"✅ Val images: {len(val_images)}")

    # Create final directories
    for split in ["train", "val"]:
        (output_dir_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("🔹 Moving files to final directories...")

    for split, split_images in [("train", train_images), ("val", val_images)]:
        for img in split_images:
            base = img.replace(".jpg", "")
            shutil.move(str(tmp_images / img), str(output_dir_path / "images" / split / img))
            shutil.move(str(tmp_labels / f"{base}.txt"), str(output_dir_path / "labels" / split / f"{base}.txt"))

    shutil.rmtree(tmp_images)
    shutil.rmtree(tmp_labels)

    print("\n✅ Dataset preparation complete and ready for YOLO training.")


if __name__ == "__main__":
    prepare_yolo_dataset(
        csv_dir="videos/soccertrack/versions/6/wide_view/annotations",
        video_dir="videos/soccertrack/versions/6/wide_view/videos",
        output_dir="videos/soccertrack/yolo_dataset_no_ball",
        train_ratio=0.8,
        frame_step=3  # Sample every 5th frame
    )
