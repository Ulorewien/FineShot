from models.yoloworld import yoloworld
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

def inference(
        image_path, 
        annotation_path, 
        dataset="refcoco", 
        confidence_threshold=0.5, 
        test=False
    ):
    image_paths = load_image_paths(image_path)
    if test:
        image_paths = image_paths[:100]
    annotations = load_annotations(annotation_path)
    
    results = {}
    i = 1
    for image_path in tqdm(image_paths, desc="Inference using YoloWorld"):
        try:
            if dataset == "refcoco":
                detections = yoloworld(image_path, annotations[str(i)]["labels"], confidence_threshold=confidence_threshold)
            else:
                detections = yoloworld(image_path, annotations[i]["labels"], confidence_threshold=confidence_threshold)
            results[i] = detections
        except:
            print(f"Error in processing image {i}")
        i += 1
    return results


if __name__ == "__main__":
    # Confidence thresholds
    cts = [0.1, 0.5, 0.9]
    for ct in cts:
        print(f"Confidence Threshold: {ct}")
        file_ct = str(ct).replace(".", "")
        
        print("Dataset: RefCOCO")
        refcoco_result_file = f"test/results/yoloworld_{file_ct}_refcoco_results.json"
        if not os.path.exists(refcoco_result_file):
            start = time()
            refcoco_results = inference(refcoco_image_path, refcoco_annotation_path, "refcoco", ct)
            end = time()
            print(f"Inference Time: {end-start} sec")
            with open(refcoco_result_file, "w") as f:
                json.dump(refcoco_results, f, indent=4)
        
        print("Dataset: VG")
        vg_result_file = f"test/results/yoloworld_{file_ct}_vg_results.json"
        if not os.path.exists(vg_result_file):
            start = time()
            vg_results = inference(vg_image_path, vg_annotation_path, "vg", ct)
            end = time()
            print(f"Inference Time: {end-start} sec")
            with open(vg_result_file, "w") as f:
                json.dump(vg_results, f, indent=4)
        
    print("Results saved successfully!")
