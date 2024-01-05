from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define path to video file
source = '/Users/cakwok/Library/Mobile Documents/com~apple~CloudDocs/HomeVideo.MOV'

# Run inference on the source
results = model(source, show=True)  
