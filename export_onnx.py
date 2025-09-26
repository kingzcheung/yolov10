from ultralytics import YOLO
# Load a model
model = YOLO("yolov10s.pt")  # load a custom trained model
# Export the model
model.export(format="onnx")