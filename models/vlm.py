import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import base64
import json
from .utils import fix_json

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True
# )

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding failed: {e}")


def qwen25_vl(image_path, class_labels):
    image = encode_image(image_path)
    class_labels = ",\n".join(class_labels)
    user_message = """
        You are an excellent object detection model who can detect any general object in the world.
        Given an image you need to find and retrun bounding box coordinates for the following labels if they are present in the image.

        labels: {class_labels}

        The output should be in the following format:
        
        {{
            "boxes":[[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
            "labels":[label1, label2, ...],
        }}
    """.format(class_labels=class_labels)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{image}",
                },
                {
                    "type": "text", 
                    "text": user_message
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=2048,
        temperature=0.1
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0].replace("```", "").replace("json", "").replace("\n", "")
    
    result = fix_json(output_text)
    return result
    

if __name__ == "__main__":
    from .utils import render_image
    
    image_path = "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg"
    class_labels = ["golden dog", "white cat", "black horse", "gorrilla", "coin"]
    detections = qwen25_vl(image_path, class_labels)
    print(detections)
    render_image(
        "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg", 
        detections["boxes"], 
        detections["labels"], 
        class_labels, 
        "/data/yashowardhan/FineShot/test/imgs/output.jpg"
    )
    