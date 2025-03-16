import torch
import base64
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolo_clip.pt").to(device)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

qwen_model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config
).to(device)
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def yolo_vlm(image_path, classes, confidence_threshold=0.5):
    predictions = yolo_model(image_path, conf=confidence_threshold)
    result = {"boxes": [], "labels": [], "scores": []}
    image = Image.open(image_path)
    objects = []
    
    for pred in predictions:
        pred = pred.cpu().numpy()
        result["boxes"] = pred.boxes.xyxy.tolist()
        result["scores"] = pred.boxes.conf.tolist()
        
        for x1, y1, x2, y2 in pred.boxes.xyxy:
            object_image = image.crop((x1, y1, x2, y2)).resize((224, 224))
            objects.append(object_image)
    
    for obj, box in zip(objects, result["boxes"]):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": obj},
                {"type": "text", "text": f"Does this object match any of these labels: {', '.join(classes)}? If not, return None."}
            ]}
        ]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(text)
        inputs = qwen_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **inputs, 
                max_new_tokens=2048,
                temperature=0.1
            )
        output_text = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        result["labels"].append(output_text[0])
    
    return result

if __name__ == "__main__":
    from models.utils import render_image
    
    classes = ["dog", "black cat"]
    image_path = "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg"
    result = yolo_vlm(image_path, classes)
    
    print(result)
    render_image(
        image_path, 
        result["boxes"], 
        result["labels"], 
        classes, 
        "/data/yashowardhan/FineShot/test/imgs/output.jpg"
    )