import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolo11l.pt").to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def yolo_clip(image_path, classes, confidence_threshold=0.5, similarity_threshold=0.5):
    predictions = yolo_model(image_path, conf=confidence_threshold, verbose=False)
    result = {"boxes": [], "labels": [], "scores": []}
    image = Image.open(image_path)
    objects, embeddings = [], []
    
    for pred in predictions:
        pred = pred.cpu().numpy()
        result["boxes"] = pred.boxes.xyxy.tolist()
        result["scores"] = pred.boxes.conf.tolist()
        
        for x1, y1, x2, y2 in pred.boxes.xyxy:
            object_image = image.crop((x1, y1, x2, y2)).resize((224, 224))
            objects.append(object_image)
            object_image = clip_processor(images=object_image, return_tensors="pt").to(device)
            with torch.no_grad():
                object_embedding = clip_model.get_image_features(pixel_values=object_image.pixel_values)
            embeddings.append(object_embedding)
    
    if objects:
        object_embeddings = torch.stack(embeddings, dim=0).squeeze(1)
        text_inputs = clip_processor(text=classes, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embeddings = clip_model.get_text_features(**text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        object_embeddings /= object_embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_embeddings @ object_embeddings.T).softmax(dim=-1)
        
        for obj_idx in range(len(objects)):
            max_sim, max_idx = similarity[:, obj_idx].max(dim=0)
            result["labels"].append(classes[max_idx] if max_sim > similarity_threshold else None)
    
    return result

if __name__ == "__main__":
    from utils import render_image
    
    classes = ["dog", "black cat"]
    image_path = "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg"
    result = yolo_clip(image_path, classes)
    
    print(result)
    render_image(
        image_path, 
        result["boxes"], 
        result["labels"], 
        classes, 
        "/data/yashowardhan/FineShot/test/imgs/output.jpg"
    )
