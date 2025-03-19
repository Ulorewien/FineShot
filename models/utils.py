"""
Output format:

{
    "image_id": {
        "boxes": [
            [x1, y1, x2, y2],
            ...
        ],
        "scores": [score1, ...],
        "labels": [label1, ...],
    }
}

Detection and Annotation format:

detections = [
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    ...
]

annotations = [
    [x1g, y1g, x2g, y2g], 
    [x1g, y1g, x2g, y2g], 
    ...
]
"""

# Utils for drawing bounding boxes and labels on images
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import re
import os

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
        print(json_string)
        print("Failed to fix JSON:", e)
        return None

def render_image(image, boxes, labels, classes, save_path=None):
    image = Image.open(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan", "magenta", "brown", "black", "white"]
    color_map = {classes[i]: colors[i%len(colors)] for i in range(len(classes))}
    index_class_map = dict(enumerate(classes))
    class_map = {v: k for k, v in index_class_map.items()}
    
    # print("color map", color_map)
    # print("index class map", index_class_map)
    # print("class map", class_map)

    for box, label in zip(boxes, labels):
        if label is None:
            continue
        x1, y1, x2, y2 = box
        if isinstance(label, int):
            box_color = color_map[index_class_map[label]]
            final_label = index_class_map[label]
        else:
            box_color = color_map[label]
            final_label = label
            
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1, y1), final_label, fill=box_color, font=font)
    
    if save_path:
        image.save(save_path)

    return image


# Function to compute IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


# Function to compute mAP
def compute_map(detections, annotations, iou_threshold=0.5):
    aps = []
    for det, ann in zip(detections, annotations):
        detected = [False] * len(det)
        true_positive = 0
        false_positive = 0
        for a in ann:
            matched = False
            for d in det:
                iou = compute_iou(d, a)
                if iou >= iou_threshold:
                    true_positive += 1
                    matched = True
                    break
            if not matched:
                false_positive += 1
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / len(ann) if len(ann) > 0 else 0
        aps.append(precision * recall)
    
    return np.mean(aps)

def load_image_paths(image_dir):
    num_images = len(os.listdir(image_dir))
    image_paths = []
    for i in range(1, num_images + 1):
        image_path = os.path.join(image_dir, f"{i}.jpg")
        image_paths.append(image_path)
    return image_paths


def load_annotations(annotation_path):
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    
    new_annotations = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in new_annotations:
            new_annotations[image_id] = {
                "boxes": [],
                "labels": []
            }
            for i in range(len(annotation["regions"])):
                new_annotations[image_id]["boxes"].append(annotation["regions"][i]["coordinates"])
                new_annotations[image_id]["labels"].append(annotation["regions"][i]["phrase"])
                
    return new_annotations


def encode_patch(patch):
    patch = np.asarray(patch)
    _, buffer = cv2.imencode(".jpeg", patch)
    return base64.b64encode(buffer).decode("utf-8")