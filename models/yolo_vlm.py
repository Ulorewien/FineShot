import cv2
import torch
import base64
from PIL import Image
from ultralytics import YOLO
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from transformers import CLIPProcessor, CLIPModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolo11n.pt").to(device)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto", 
    quantization_config=quantization_config
).to(device)
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_object_patches(image_path):
    objects = []
    image = Image.open(image_path)
    results = yolo_model.predict(image_path)

    for result in results:
        xyxy = result.boxes.xyxy
 
    for x1, y1, x2, y2 in xyxy:
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        object_image = image.crop((x1, y1, x2, y2))
        object_image = object_image.resize((224, 224))
        objects.append(object_image)

    return objects

def encode_patch(patch):
    _, buffer = cv2.imencode(".jpeg", patch)
    return base64.b64encode(buffer).decode("utf-8")

def get_patch_descriptions(patches):
    patch_descriptions = []
    for patch in patches:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{encode_patch(patch)}",
                    },
                    {
                        "type": "text", 
                        "text": "Give a detailed description of this image."
                    },
                ],
            }
        ]
        
        text = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(text)
        inputs = qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(qwen_model.device)

        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        patch_descriptions.append(output_text)
        
    return patch_descriptions

def get_clip_similarity(text, patch_descriptions, threshold=0.5):
    text_input = clip_processor(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_input)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    similarities = []
    for description in patch_descriptions:
        object_input = clip_processor(description, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            object_embeddings = clip_model.get_text_features(**object_input)
        object_embeddings /= object_embeddings.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * text_embeddings @ object_embeddings.T).softmax(dim=-1)
        max_similarity, max_index = similarity.max(dim=1)
        
        print(max_similarity)
        
        if max_similarity > threshold:
            return similarities[max_index]
    
    return None

image_path = "test2.jpeg" # Change this to the path of your image
text = "test" # Change this to your text
patches = get_object_patches(image_path)
patch_descriptions = get_patch_descriptions(patches)
print("Patch descriptions:", patch_descriptions)

detected_patch = get_clip_similarity(text, patch_descriptions)
if detected_patch:
    detected_patch.show()