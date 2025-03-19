from models.yolo_clip_vlm import yolo_clip_vlm
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
        similarity_threshold=0.5,
        test = False
    ):
    image_paths = load_image_paths(image_path)
    if test:
        image_paths = image_paths[45:55]
    annotations = load_annotations(annotation_path)
    
    results = {}
    i = 1
    for image_path in tqdm(image_paths, desc="Inference using YOLO-CLIP-VLM"):
        try:
            if dataset == "refcoco":
                detections = yolo_clip_vlm(
                    image_path, 
                    annotations[str(i)]["labels"], 
                    confidence_threshold=confidence_threshold,
                    similarity_threshold=similarity_threshold
                )
            else:
                detections = yolo_clip_vlm(
                    image_path, 
                    annotations[i]["labels"], 
                    confidence_threshold=confidence_threshold,
                    similarity_threshold=similarity_threshold
                )
            results[i] = detections
        except:
            print(f"Error in processing image {i}")
        i += 1
    return results
    

if __name__ == "__main__":
    cts = [0.1, 0.5, 0.9]
    sts = [0.25, 0.5, 0.75]
    refcoco_speed = []
    vg_speed = []
    
    for ct in cts:
        file_ct = str(ct).replace(".", "")
        for st in sts:
            print(f"Running for confidence threshold: {ct} and similarity threshold: {st}")
            file_st = str(st).replace(".", "")
            
            print("Dataset: RefCOCO")
            refcoco_result_file = f"test/results/yolo_clip_vlm_{file_ct}_{file_st}_refcoco_results.json"
            if not os.path.exists(refcoco_result_file):
                start = time()
                refcoco_results = inference(refcoco_image_path, refcoco_annotation_path, "refcoco", ct, st)
                end = time()
                refcoco_speed.append(end - start)
                with open(refcoco_result_file, "w") as f:
                    json.dump(refcoco_results, f, indent=4)
            
            print("Dataset: VG")
            vg_result_file = f"test/results/yolo_clip_vlm_{file_ct}_{file_st}_vg_results.json"
            if not os.path.exists(vg_result_file):
                start = time()
                vg_results = inference(vg_image_path, vg_annotation_path, "vg", ct, st)
                end = time()
                vg_speed.append(end - start)
                with open(vg_result_file, "w") as f:
                    json.dump(vg_results, f, indent=4)
    
    print("RefCOCO Speeds:", refcoco_speed)
    print("Visual Genome Speeds:", vg_speed)
    print("Results saved successfully!")