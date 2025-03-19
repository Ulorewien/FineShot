import torch
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
# from .utils import encode_patch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolo11l.pt").to(device)

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
)
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def encode_patch(patch):
    patch = np.asarray(patch)
    _, buffer = cv2.imencode(".jpeg", patch)
    return base64.b64encode(buffer).decode("utf-8")

def yolo_vlm(image_path, classes, confidence_threshold=0.5):
    predictions = yolo_model(image_path, conf=confidence_threshold, verbose=False)
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
        encoded_image = encode_patch(obj)
        labels_string = '\n'.join(classes)
        messages = [
            {"role": "user", "content": [
                {
                    "type": "image", 
                    "image": f"data:image/jpeg;base64,{encoded_image}"
                },
                {
                    "type": "text", 
                    "text": f"LABELS: {labels_string} \n\nSelect the best label for the given image. If no label matches the image, return None. Only return one label without any other text."
                }
            ]
            }
        ]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        
        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.1
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if "None" not in output_text[0]:
            result["labels"].append(output_text[0])
        else:
            result["labels"].append(None)
    
    return result

if __name__ == "__main__":
    from utils import render_image
    
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