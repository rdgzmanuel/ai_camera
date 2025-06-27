import os
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.jit import RecursiveScriptModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision as TorchMetricsMAP
from PIL import Image
from typing import Tuple, List, Dict


class FootballDetectionDataset(Dataset):
    """
    YOLO-style dataset for soccer video object detection.
    """

    def __init__(self, image_dir: str, label_dir: str, transform: transforms.Compose = None) -> None:
        """
        Constructor for SoccerDetectionDataset.

        Args:
            image_dir (str): Path to directory containing images.
            label_dir (str): Path to directory containing YOLO-format labels.
            transform (Compose, optional): Torchvision image transform pipeline.
        """
        self.image_dir: str = image_dir
        self.label_dir: str = label_dir
        self.transform: transforms.Compose = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

        self.image_files: List[str] = [
            f for f in os.listdir(image_dir) if f.endswith(".jpg")
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_file: str = self.image_files[index]
        image_path: str = os.path.join(self.image_dir, image_file)
        label_path: str = os.path.join(self.label_dir, image_file.replace(".jpg", ".txt"))

        image: Image.Image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        boxes: List[List[float]] = []
        labels: List[int] = []

        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        labels.append(int(cls))
                        boxes.append([x, y, w, h])

        targets: Dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),  # [N, 4]
            "labels": torch.tensor(labels, dtype=torch.int64)   # [N]
        }

        return image_tensor, targets


def load_football_data(
    dataset_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares DataLoaders for YOLO-style football object detection dataset.

    Args:
        dataset_path (str): Root path to football dataset (should contain /images and /labels).
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        train (bool): Whether to return training + val or test loader.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train, val, test dataloaders.
    """
    image_dir: Path = os.path.join(dataset_path, "images")
    label_dir: Path = os.path.join(dataset_path, "labels")

    full_dataset: FootballDetectionDataset = FootballDetectionDataset(image_dir=image_dir, label_dir=label_dir)

    train_loader = val_loader = test_loader = None

    if train:
        train_len: int = int(0.8 * len(full_dataset))
        val_len: int = len(full_dataset) - train_len
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

        train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader: DataLoader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def convert_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Converts absolute bounding box coordinates to normalized YOLO format.

    Args:
        x (float): Top-left x coordinate.
        y (float): Top-left y coordinate.
        w (float): Width of the box.
        h (float): Height of the box.
        img_w (int): Image width.
        img_h (int): Image height.

    Returns:
        tuple[float, float, float, float]: Normalized (x_center, y_center, width, height).
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h


def convert_annotations(csv_dir: Path, video_dir: Path, output_dir: Path) -> None:
    """
    Converts soccer CSV annotations to YOLO format labels and saves corresponding video frames.

    Args:
        csv_dir (Path): Path to directory containing CSV annotation files.
        video_dir (Path): Path to directory containing video files.
        output_dir (Path): Path to output root directory for YOLO dataset.
    """
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_dir.glob("*.csv"):
        video_file = video_dir / (csv_file.stem + ".mp4")
        if not video_file.exists():
            print(f"Warning: Video not found for {csv_file.name}")
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Error: Failed to open video {video_file}")
            continue

        df = pd.read_csv(csv_file, header=None)
        object_ids = df.iloc[0].tolist()
        data = df.iloc[2:].reset_index(drop=True)

        saved_frames = 0

        for frame_idx, row in data.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            img_h, img_w = frame.shape[:2]
            image_name = f"{csv_file.stem}_frame{frame_idx:04d}.jpg"
            label_name = image_name.replace(".jpg", ".txt")

            image_path = images_dir / image_name
            label_path = labels_dir / label_name

            cv2.imwrite(str(image_path), frame)

            with open(label_path, "w") as f:
                for col in range(1, len(row) - 3, 4):
                    obj_id = object_ids[col]
                    try:
                        h = float(row[col])
                        x = float(row[col + 1])
                        y = float(row[col + 2])
                        w = float(row[col + 3])
                    except (ValueError, IndexError):
                        continue

                    if w <= 0 or h <= 0:
                        continue

                    cls = 1 if "BALL" in str(obj_id).upper() else 0
                    x_center, y_center, w_norm, h_norm = convert_to_yolo(x, y, w, h, img_w, img_h)
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            saved_frames += 1

        cap.release()

        print(f"Processed {csv_file.name} — {saved_frames} frames saved")



class MeanAveragePrecision:
    """
    Thin wrapper around torchmetrics' MeanAveragePrecision for YOLO output format.
    """

    def __init__(self) -> None:
        self.metric = TorchMetricsMAP(iou_type="bbox", class_metrics=False)

    def update(self, predictions: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]) -> None:
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            predictions: List of predictions with keys 'boxes', 'scores', 'labels'.
            targets: Ground truth with same keys (except scores).
        """
        self.metric.update(predictions, targets)

    def compute(self) -> float:
        """
        Computes the final mAP value (mAP@0.5).

        Returns:
            float: mAP@0.5 score.
        """
        result = self.metric.compute()
        return result["map_50"].item()  # mAP@IoU=0.5
    

def process_yolo_outputs(outputs: torch.Tensor, conf_threshold: float = 0.3) -> list[dict[str, torch.Tensor]]:
    """
    Processes raw YOLO outputs to the format expected by torchmetrics MeanAveragePrecision.

    Args:
        outputs (Tensor): Raw model output of shape [B, N, 5 + C] in xywh format.
        conf_threshold (float): Confidence threshold to filter predictions.

    Returns:
        list[dict[str, Tensor]]: List of dictionaries per image.
    """
    batch_size: int = outputs.shape[0]
    processed: list[dict[str, torch.Tensor]] = []

    for i in range(batch_size):
        output = outputs[i]  # [N, 5 + C]
        pred_boxes = output[..., :4]
        conf = output[..., 4]
        class_logits = output[..., 5:]

        class_scores, class_ids = torch.max(class_logits, dim=-1)
        scores = conf * class_scores.sigmoid()

        # Filter by confidence threshold
        keep = scores > conf_threshold
        if keep.sum() == 0:
            processed.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64)
            })
            continue

        boxes_xywh = pred_boxes[keep]
        scores_kept = scores[keep]
        labels_kept = class_ids[keep]

        # Convert to xyxy
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:]
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)

        processed.append({
            "boxes": xyxy.detach().cpu(),
            "scores": scores_kept.detach().cpu(),
            "labels": labels_kept.detach().cpu()
        })

    return processed


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed (int): seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    Saves a YOLOv11 model as a TorchScript file on the CPU.

    Args:
        model (torch.nn.Module): Trained model to save.
        name (str): Name of the model file (without extension).
    """
    if not os.path.isdir("models"):
        os.makedirs("models")

    model_cpu: torch.nn.Module = model.to("cpu").eval()

    try:
        # Script the model for portability (TorchScript)
        model_scripted: RecursiveScriptModule = torch.jit.script(model_cpu)
    except Exception as e:
        raise RuntimeError(f"Failed to script model: {e}")

    model_scripted.save(f"models/{name}.pt")


def load_model(name: str) -> RecursiveScriptModule:
    """
    Loads a TorchScript model from the models directory.

    Args:
        name (str): Filename of the saved model (without extension).

    Returns:
        RecursiveScriptModule: Loaded model.
    """
    local_path: str = f"models/{name}.pt"
    fallback_path: str = f"/workspace/project/models/{name}.pt"  # for workspace support

    model_path: str = local_path if os.path.exists(local_path) else fallback_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model: RecursiveScriptModule = torch.jit.load(model_path, map_location="cpu")
    return model



if __name__ == "__main__":
    csv_dir = Path("./videos/soccertrack/wide_view/annotations")
    video_dir = Path("./videos/soccertrack/wide_view/videos")
    output_dir = Path("./videos/soccertrack/yolo_dataset")

    convert_annotations(csv_dir, video_dir, output_dir)
