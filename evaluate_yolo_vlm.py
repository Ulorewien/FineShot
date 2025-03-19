from models.yolo_vlm import yolo_vlm
from models.utils import *
import json
import os
from time import time
import torch
from tqdm import tqdm

refcoco_image_path = "/data/yashowardhan/FineShot/test/refcoco_images"
refcoco_annotation_path = "/data/yashowardhan/FineShot/data/refcoco data/refcoco_updated.json"

vg_image_path = "/data/yashowardhan/FineShot/test/vg_images"
vg_annotation_path = "/data/yashowardhan/FineShot/data/visual genome data/vg_subset.json"

def inference(
        image_path, 
        annotation_path, 
        dataset="refcoco", 
        confidence_threshold=0.5, 
        test = False
    ):
    image_paths = load_image_paths(image_path)
    if test:
        image_paths = image_paths[45:55]
    annotations = load_annotations(annotation_path)
    
    results = {}
    i = 1
    for image_path in tqdm(image_paths, desc="Inference using YOLO-VLM"):
        try:
            if dataset == "refcoco":
                detections = yolo_vlm(image_path, annotations[str(i)]["labels"], confidence_threshold=confidence_threshold)
            else:
                detections = yolo_vlm(image_path, annotations[i]["labels"], confidence_threshold=confidence_threshold)
            results[i] = detections
        except:
            print(f"Error in processing image {i}")
        i += 1
    return results
    

if __name__ == "__main__":
    # Confidence thresholds
    cts = [0.1, 0.5, 0.9]
    refcoco_speed = []
    vg_speed = []
    
    for ct in cts:
        file_ct = str(ct).replace(".", "")
        print(f"Confidence threshold: {ct}")
        
        print("Dataset: RefCOCO")
        refcoco_result_file = f"/data/yashowardhan/FineShot/test/results/yolo_vlm_{file_ct}_refcoco_results.json"
        if os.path.exists(refcoco_result_file):
            start = time()
            refcoco_results = inference(refcoco_image_path, refcoco_annotation_path, "refcoco", ct)
            end = time()
            refcoco_speed.append(end-start)
            with open(refcoco_result_file, "w") as f:
                json.dump(refcoco_results, f, indent=4)
        
        print("Dataset: VG")
        vg_result_file = f"/data/yashowardhan/FineShot/test/results/yolo_vlm_{file_ct}_vg_results.json"
        if os.path.exists(vg_result_file):
            start = time()
            vg_results = inference(vg_image_path, vg_annotation_path, "vg", ct)
            end = time()
            vg_speed.append(end-start)
            with open(vg_result_file, "w") as f:
                json.dump(vg_results, f, indent=4)
    
    print("RefCOCO Speeds:", refcoco_speed)
    print("VG Speeds:", vg_speed)
    print("Results saved successfully!")