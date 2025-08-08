import os
import random
import shutil

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    for split, split_images in [("train", train_images), ("val", val_images)]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

        for img in split_images:
            base = os.path.splitext(img)[0]
            shutil.copy(os.path.join(images_dir, img), os.path.join(output_dir, "images", split, img))
            label_path = os.path.join(labels_dir, base + ".txt")
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_dir, "labels", split, base + ".txt"))

split_dataset(
    images_dir="videos/soccertrack/yolo_dataset/images",
    labels_dir="videos/soccertrack/yolo_dataset/labels",
    output_dir="videos/soccertrack/yolo_dataset"
)