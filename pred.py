import torch
from ultralytics import YOLO
import os

# Check if model file exists
model_path = "yolov10n.pt"

    # Load a model
model = YOLO(model_path)  # load a custom trained model


one = torch.ones(1, 3, 640, 640)
# 定义钩子函数
def hook(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output}")
    return output


layer = model.model.model[0]  # type: ignore # model.model是实际的网络结构
handle = layer.register_forward_hook(hook)
results = model(one)
    
    