from models.yolo_clip import yolo_clip
from models.utils import *
import json
import torch
from tqdm import tqdm

refcoco_image_path = "/data/yashowardhan/FineShot/test/refcoco_images"
refcoco_annotation_path = "/data/yashowardhan/FineShot/data/refcoco data/refcoco_updated.json"

vg_image_path = "/data/yashowardhan/FineShot/test/vg_images"
vg_annotation_path = "/data/yashowardhan/FineShot/data/visual genome data/vg_subset.json"

def inference(image_path, annotation_path, dataset="refcoco"):
    image_paths = load_image_paths(image_path)
    image_paths = image_paths[:5]
    annotations = load_annotations(annotation_path)
    
    results = {}
    i = 1
    for image_path in tqdm(image_paths, desc="Inference using YOLO-CLIP"):
        try:
            if dataset == "refcoco":
                detections = yolo_clip(image_path, annotations[str(i)]["labels"], confidence_threshold=0.5)
            else:
                detections = yolo_clip(image_path, annotations[i]["labels"], confidence_threshold=0.5)
            results[i] = detections
        except:
            print(f"Error in processing image {i}")
        i += 1
    return results
    

if __name__ == "__main__":
    refcoco_results = inference(refcoco_image_path, refcoco_annotation_path, "refcoco")
    with open("/data/yashowardhan/FineShot/test/results/yolo_clip_refcoco_results.json", "w") as f:
        json.dump(refcoco_results, f, indent=4)
    
    vg_results = inference(vg_image_path, vg_annotation_path, "vg")
    with open("/data/yashowardhan/FineShot/test/results/yolo_clip_vg_results.json", "w") as f:
        json.dump(vg_results, f, indent=4)
    
    print("Results saved successfully!")