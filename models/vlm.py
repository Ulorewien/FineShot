"""
Prompt:

You are an excellent object detection model who can detect any general object in the world.
Given an image you need to find and retrun bounding box coordinates for the following labels if they are present in the image.

labels: dog, cat, horse monkey, coin

The output should be in the following format:

{
"boxes":[[x1, y1, x2, y2]],
"labels":[label1],
}

"""

import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


def qwen25_vl(image_path, class_labels):
    pass

if __name__ == "__main__":
    image_path = "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg"
    class_labels = ["dog", "cat", "horse", "monkey", "coin"]
    