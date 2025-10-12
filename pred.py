from ultralytics import YOLO
import os

# Check if model file exists
model_path = "yolov10s.pt"

    # Load a model
model = YOLO(model_path)  # load a custom trained model

results = model("./testdata/bus.jpg")
    
    