from ultralytics import YOLO
import os

# Check if model file exists
model_path = "yolov10s.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} does not exist")
    exit(1)

try:
    # Load a model
    model = YOLO(model_path)  # load a custom trained model
    
    # Export the model with additional parameters
    model.export(format="onnx", 
                 imgsz=640,           # specify image size
                 simplify=True,       # simplify model
                 opset=11)            # specify opset version
    print("ONNX model exported successfully!")
except Exception as e:
    print(f"Error occurred during export: {e}")