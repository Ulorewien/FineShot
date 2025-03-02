# Utils for drawing bounding boxes and labels on images
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import re

def fix_json(json_string):
    # Fix trailing commas
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*\]', ']', json_string)

    # Replace single quotes with double quotes
    json_string = re.sub(r"'", '"', json_string)

    # Add double quotes around unquoted keys
    json_string = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_string)

    # Validate and load the JSON
    try:
        data = json.loads(json_string)
        print("JSON is valid after fixing!")
        return data  # Return pretty-printed JSON
    except json.JSONDecodeError as e:
        print("Failed to fix JSON:", e)
        return None

def render_image(image, boxes, labels, classes, save_path=None):
    image = Image.open(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan", "magenta", "brown"]
    color_map = dict(zip(set(labels), colors))
    index_class_map = dict(enumerate(classes))

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        if isinstance(label, int):
            box_color = color_map[index_class_map[label]]
            final_label = class_map[label]
        else:
            box_color = color_map[label]
            final_label = label
            
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
        draw.text((x1, y1), final_label, fill=box_color, font=font)
    
    if save_path:
        image.save(save_path)

    return image