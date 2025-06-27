import torch

ckpt = torch.load("models/yolox_m.pth", map_location="cpu", weights_only=False)
print(type(ckpt))
