import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from tqdm.auto import tqdm

from src.models import YOLOv11FineTuner  # custom fine-tuner class
from src.utils import (
    load_football_data,  # custom dataset loader returning (image, target)
    save_model,
    set_seed,
    MeanAveragePrecision,
    process_yolo_outputs
)
from src.loss import YOLOLoss

device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)
torch.set_num_threads(8)

DATA_PATH: str = "videos/soccertrack/yolo_dataset"
NUM_CLASSES: int = 2  # player, ball

def main() -> None:
    """
    Main function to train YOLOv11 model on football dataset.
    """
    # Hyperparameters
    epochs: int = 2
    batch_size: int = 16
    lr: float = 1e-3
    step_size: int = 20
    gamma: float = 0.5
    name: str = "yolov11_1"
    backbone_model: str = "models/yolo11m.pt"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    train_loader: DataLoader
    val_loader: DataLoader
    train_loader, val_loader, _ = load_football_data(DATA_PATH, batch_size=batch_size)

    model: YOLOv11FineTuner = YOLOv11FineTuner(model_path=backbone_model, num_classes=NUM_CLASSES).to(device)

    criterion: YOLOLoss = YOLOLoss(num_classes=NUM_CLASSES)

    optimizer: Adam = Adam(model.parameters(), lr=lr)
    scheduler: StepLR = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(epochs)):
        train_step(model, train_loader, criterion, optimizer, writer, epoch, device)
        val_step(model, val_loader, criterion, writer, epoch, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs} completed.")

    save_model(model, name)

    return None


def train_step(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    Perform one training epoch.

    Args:
        model: YOLO model instance.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number.
        device: Computation device.
    """
    model.train()
    losses: list[float] = []

    for images, targets in train_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    writer.add_scalar("train/loss", np.mean(losses), epoch)


def val_step(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    Perform one validation epoch.

    Args:
        model: YOLO model instance.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number.
        device: Computation device.
    """
    model.eval()
    losses: list[float] = []
    map_metric: MeanAveragePrecision = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            loss = criterion(outputs, targets)
            losses.append(loss.item())

            predictions: list[dict[str, torch.Tensor]] = process_yolo_outputs(outputs, NUM_CLASSES)

            map_metric.update(predictions, targets)

    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/mAP", map_metric.compute(), epoch)


if __name__ == "__main__":
    main()