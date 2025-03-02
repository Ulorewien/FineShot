from ultralytics import YOLOWorld
import torch
from utils import render_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOWorld("yolov8l-worldv2.pt") 
model = model.to(device)
model.eval()

def yoloworld(image_path, classes, confidence_threshold=0.2):
    model.set_classes(classes)
    predictions = model(image_path, conf=confidence_threshold)
    result = dict()

    for pred in predictions:
        pred = pred.cpu().numpy()
        result["boxes"] = pred.boxes.xyxy.tolist()
        result["labels"] = pred.boxes.cls.tolist()
        result["scores"] = pred.boxes.conf.tolist()
    
    return result

if __name__ == "__main__":
    classes = ["dog", "black cat"]
    image_path = "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg"
    result = yoloworld(image_path, classes)
    print(result)
    render_image(
        image_path, 
        result["boxes"], 
        result["labels"], 
        classes, 
        "/data/yashowardhan/FineShot/test/imgs/output.jpg"
    )