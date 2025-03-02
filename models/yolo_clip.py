import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolo11n.pt").to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_object_embeddings(image_path):
    embeddings, objects = [], []
    image = Image.open(image_path)
    results = yolo_model.predict(image_path)

    for result in results:
        xyxy = result.boxes.xyxy
 
    for x1, y1, x2, y2 in xyxy:
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        object_image = image.crop((x1, y1, x2, y2))
        object_image = object_image.resize((224, 224))
        objects.append(object_image)
        object_image = clip_processor(images=object_image, return_tensors="pt").to(device)
        with torch.no_grad():
            object_embedding = clip_model.get_image_features(pixel_values=object_image.pixel_values)
        embeddings.append(object_embedding)

    return torch.stack(embeddings, dim=0), objects

def get_object(object_embeddings, objects, text, threshold=0.5):
    text_input = clip_processor(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_input)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    object_embeddings = object_embeddings.to(device)
    object_embeddings /= object_embeddings.norm(dim=-1, keepdim=True)
    
    similarity = (100.0 * text_embeddings @ object_embeddings.T).softmax(dim=-1)
    max_similarity, max_index = similarity.max(dim=1)
    
    if max_similarity > threshold:
        return objects[max_index]
    
    return None

image_path = "test2.jpeg" # Change this to the path of your image
text = "test" # Change this to your text

object_embeddings, objects = get_object_embeddings(image_path)
object_embeddings = object_embeddings.squeeze(1)
print("CLIP embeddings:", object_embeddings)
print("Embeddings shape:", object_embeddings.shape)

detected_object = get_object(object_embeddings, objects, text)
if detected_object:
    detected_object.show()