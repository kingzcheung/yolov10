from ultralytics import YOLO # type: ignore
import os

# Check if model file exists
model_path = "yolov10x.pt"

try:
    # Load a model
    model = YOLO(model_path)  # load a custom trained model
    
    print(model.model)
    
    # Export the model with additional parameters
    model.export(format="onnx", 
                 imgsz=640,           # specify image size
                 simplify=True,       # simplify model
                 opset=11)            # specify opset version
    print("ONNX model exported successfully!")
except Exception as e:
    print(f"Error occurred during export: {e}")