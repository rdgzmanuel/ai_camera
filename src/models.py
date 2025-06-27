import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOv11FineTuner(nn.Module):
    """
    YOLOv11 fine-tuning model for object detection (e.g., football players and ball).
    """

    def __init__(self, model_path: str = "models/yolo11n.pt", num_classes: int = 2, device: str = "cpu") -> None:
        """
        Constructor for YOLOv11FineTuner.

        Args:
            num_classes (int): Number of detection classes (e.g., 2 for player and ball).
            pretrained (bool): Whether to load pretrained weights from COCO.
        """
        super().__init__()

        self.device: str = device

        self.model: YOLO = YOLO(model_path).to(device)

        # Replace the detection head for custom number of classes
        self._replace_head(num_classes)

    def _replace_head(self, num_classes: int) -> None:
        """
        Replace the YOLO detection head with one that matches the number of output classes.

        Args:
            num_classes (int): Number of detection classes to output.
        """
        # Access the detection head depending on architecture
        # For YOLOv11, head might be a custom module with conv layers
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'detect'):
            old_head = self.model.head.detect
            in_channels: int = old_head[0].in_channels

            self.model.head.detect = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, num_classes * (5 + 1), kernel_size=1)  # 5 box coords + 1 conf
            )
        else:
            raise ValueError("Could not locate the detection head in YOLOv11 model.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through YOLOv11 model.

        Args:
            x (Tensor): Input batch of images [batch, channels, height, width].

        Returns:
            Tensor: Output predictions (usually multi-scale feature maps).
        """
        return self.model(x)
