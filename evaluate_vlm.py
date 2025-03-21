from models.vlm import qwen25_vl
from models.utils import *
import json
import os
from time import time
import torch
from tqdm import tqdm

refcoco_image_path = "test/refcoco_images"
refcoco_annotation_path = "data/refcoco data/refcoco_updated.json"

vg_image_path = "test/vg_images"
vg_annotation_path = "data/visual genome data/vg_subset.json"

def inference(image_path, annotation_path, dataset="refcoco", test=False):
    image_paths = load_image_paths(image_path)
    if test:
        image_paths = image_paths[45:55]
    annotations = load_annotations(annotation_path)
    
    results = {}
    i = 1
    for image_path in tqdm(image_paths, desc="Inference using Qwen2.5-VL"):
        if dataset == "refcoco":
            detections = qwen25_vl(image_path, annotations[str(i)]["labels"])
        else:
            detections = qwen25_vl(image_path, annotations[i]["labels"])
        results[i] = detections
        i += 1
    return results


if __name__ == "__main__":
    print("Dataset: RefCOCO")
    start = time()
    refcoco_results = inference(refcoco_image_path, refcoco_annotation_path, "refcoco")
    end = time()
    print(f"Inference Time: {end-start} sec")
    with open("test/results/vlm_refcoco_results.json", "w") as f:
        json.dump(refcoco_results, f, indent=4)
    
    print("Dataset: VG")
    start = time()
    vg_results = inference(vg_image_path, vg_annotation_path, "vg")
    end = time()
    print(f"Inference Time: {end-start} sec")
    with open("test/results/vlm_vg_results.json", "w") as f:
        json.dump(vg_results, f, indent=4)
        
    print("Results saved successfully!")
        
