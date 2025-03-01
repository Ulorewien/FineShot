import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolov8n.pt").to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def detect_objects_and_get_clip_embeddings(image_path):
    image = Image.open(image_path)

    results = yolo_model.predict(image_path)

    embeddings = []
    for result in results:
        xyxy = result.boxes.xyxy
 
    for x1, y1, x2, y2 in xyxy:
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        object_image = image.crop((x1, y1, x2, y2))
        object_image = clip_processor(images=object_image, return_tensors="pt").to(device)
        with torch.no_grad():
            object_embedding = clip_model.get_image_features(pixel_values=object_image.pixel_values)
        embeddings.append(object_embedding)

    return torch.stack(embeddings, dim=0)

image_path = "test2.jpeg" # Change this to the path of your image
object_embeddings = detect_objects_and_get_clip_embeddings(image_path).squeeze(1)
print("CLIP embeddings:", object_embeddings)
print("Embeddings shape:", object_embeddings.shape)