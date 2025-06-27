import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    Custom YOLOv11-style object detection loss combining:
    - CIoU loss for bounding boxes
    - BCE loss for objectness score
    - BCE loss for class predictions
    """

    def __init__(self, num_classes: int, box_weight: float = 0.05, obj_weight: float = 1.0, cls_weight: float = 0.5) -> None:
        """
        Initializes the YOLOLoss class.

        Args:
            num_classes (int): Number of classes in the dataset.
            box_weight (float): Weight for the CIoU loss.
            obj_weight (float): Weight for the objectness loss.
            cls_weight (float): Weight for the classification loss.
        """
        super().__init__()
        self.num_classes: int = num_classes
        self.bce_obj: nn.Module = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_cls: nn.Module = nn.BCEWithLogitsLoss(reduction="mean")
        self.box_weight: float = box_weight
        self.obj_weight: float = obj_weight
        self.cls_weight: float = cls_weight


    def forward(self, predictions: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the total loss for YOLO object detection.

        Args:
            predictions (torch.Tensor): Model outputs of shape [B, A, S, S, 5 + num_classes].
            targets (list[dict[str, torch.Tensor]]): Ground-truth boxes and labels per image.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Total loss and individual components.
        """
        batch_size: int = predictions.size(0)
        device: torch.device = predictions.device

        preds: torch.Tensor = predictions.view(batch_size, -1, 5 + self.num_classes)
        pred_boxes: torch.Tensor = preds[..., :4]
        pred_obj: torch.Tensor = preds[..., 4]
        pred_cls: torch.Tensor = preds[..., 5:]

        true_boxes: torch.Tensor = torch.zeros_like(pred_boxes).to(device)
        true_obj: torch.Tensor = torch.zeros_like(pred_obj).to(device)
        true_cls: torch.Tensor = torch.zeros_like(pred_cls).to(device)

        for i in range(batch_size):
            t = targets[i]
            num_targets: int = len(t["boxes"])
            if num_targets == 0:
                continue
            true_obj[i, :num_targets] = 1.0
            true_boxes[i, :num_targets] = t["boxes"]
            for j in range(num_targets):
                true_cls[i, j, t["labels"][j]] = 1.0

        ciou_loss: torch.Tensor = 1.0 - self.ciou(pred_boxes, true_boxes).mean()
        obj_loss: torch.Tensor = self.bce_obj(pred_obj, true_obj)
        cls_loss: torch.Tensor = self.bce_cls(pred_cls, true_cls)

        total_loss: torch.Tensor = (
            self.box_weight * ciou_loss +
            self.obj_weight * obj_loss +
            self.cls_weight * cls_loss
        )

        loss_components: dict[str, torch.Tensor] = {
            "ciou": ciou_loss.detach(),
            "obj": obj_loss.detach(),
            "cls": cls_loss.detach(),
            "total": total_loss.detach()
        }

        return total_loss, loss_components


    def ciou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Computes Complete IoU (CIoU) between predicted and target boxes.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes [B, N, 4] in xywh format.
            target_boxes (torch.Tensor): Target boxes [B, N, 4] in xywh format.

        Returns:
            torch.Tensor: CIoU values for each predicted-target pair.
        """
        pred_xy: torch.Tensor = pred_boxes[..., :2]
        pred_wh: torch.Tensor = pred_boxes[..., 2:]
        target_xy: torch.Tensor = target_boxes[..., :2]
        target_wh: torch.Tensor = target_boxes[..., 2:]

        pred_min: torch.Tensor = pred_xy - pred_wh / 2
        pred_max: torch.Tensor = pred_xy + pred_wh / 2
        target_min: torch.Tensor = target_xy - target_wh / 2
        target_max: torch.Tensor = target_xy + target_wh / 2

        inter_min: torch.Tensor = torch.max(pred_min, target_min)
        inter_max: torch.Tensor = torch.min(pred_max, target_max)
        inter: torch.Tensor = (inter_max - inter_min).clamp(min=0)
        inter_area: torch.Tensor = inter[..., 0] * inter[..., 1]

        pred_area: torch.Tensor = pred_wh[..., 0] * pred_wh[..., 1]
        target_area: torch.Tensor = target_wh[..., 0] * target_wh[..., 1]
        union_area: torch.Tensor = pred_area + target_area - inter_area + 1e-6

        iou: torch.Tensor = inter_area / union_area

        center_dist: torch.Tensor = torch.sum((pred_xy - target_xy) ** 2, dim=-1)

        enc_min: torch.Tensor = torch.min(pred_min, target_min)
        enc_max: torch.Tensor = torch.max(pred_max, target_max)
        enc_diag: torch.Tensor = torch.sum((enc_max - enc_min) ** 2, dim=-1) + 1e-6

        v: torch.Tensor = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(pred_wh[..., 0] / (pred_wh[..., 1] + 1e-6)) -
            torch.atan(target_wh[..., 0] / (target_wh[..., 1] + 1e-6)), 2
        )
        alpha: torch.Tensor = v / (1 - iou + v + 1e-6)

        ciou: torch.Tensor = iou - (center_dist / enc_diag) - alpha * v
        return ciou.clamp(min=-1.0, max=1.0)
