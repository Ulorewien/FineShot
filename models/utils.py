# Utils for drawing bounding boxes and labels on images
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def render_image(image, boxes, labels, classes, save_path=None):
    image = Image.open(image)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan", "magenta", "brown"]
    color_map = dict(zip(set(labels), colors))
    class_map = dict(enumerate(classes))

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color_map[label], width=1)
        draw.text((x1, y1), class_map[label], fill=color_map[label], font=font)
    
    if save_path:
        image.save(save_path)

    return image