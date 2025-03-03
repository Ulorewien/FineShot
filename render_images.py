from models.utils import *
from tqdm import tqdm
import json

refcoco_image_path = "/data/yashowardhan/FineShot/test/refcoco_images"
refcoco_annotation_path = "/data/yashowardhan/FineShot/data/refcoco data/refcoco_updated.json"

vg_image_path = "/data/yashowardhan/FineShot/test/vg_images"
vg_annotation_path = "/data/yashowardhan/FineShot/data/visual genome data/vg_subset.json"


def draw_boxes(image_path, detection_path, annotation_path, dataset="refcoco", model="owlv2"):
    image_paths = load_image_paths(image_path)
    image_paths = image_paths[:5]
    annotations = load_annotations(annotation_path)
    
    with open(detection_path, "r") as f:
        detections = json.load(f)
    
    i = 1
    for image_path in tqdm(image_paths, desc="Drawing boxes"):
        try:
            if dataset == "refcoco":
                class_labels = annotations[str(i)]["labels"]
            else:
                class_labels = annotations[i]["labels"]
                
            detection = detections[str(i)]    
            
            render_image(
                image_path, 
                detection["boxes"], 
                detection["labels"], 
                class_labels, 
                f"/data/yashowardhan/FineShot/test/{dataset}_results/{model}_{i}.jpg"
            )
        except:
            print(f"Failed to render image {i}")
        i += 1
        

if __name__ == "__main__":
    draw_boxes(
        refcoco_image_path, 
        "/data/yashowardhan/FineShot/test/results/owlv2_refcoco_results.json", 
        refcoco_annotation_path, 
        "refcoco", 
        "owlv2"
    )
    draw_boxes(
        vg_image_path, 
        "/data/yashowardhan/FineShot/test/results/owlv2_vg_results.json", 
        vg_annotation_path,
        "vg", 
        "owlv2"
    )
    
    draw_boxes(
        refcoco_image_path, 
        "/data/yashowardhan/FineShot/test/results/yoloworld_refcoco_results.json", 
        refcoco_annotation_path, 
        "refcoco", 
        "yoloworld"
    )
    draw_boxes(
        vg_image_path, 
        "/data/yashowardhan/FineShot/test/results/yoloworld_vg_results.json", 
        vg_annotation_path,
        "vg", 
        "yoloworld"
    )
    
    draw_boxes(
        refcoco_image_path, 
        "/data/yashowardhan/FineShot/test/results/vlm_refcoco_results.json", 
        refcoco_annotation_path, 
        "refcoco", 
        "vlm"
    )   
    draw_boxes(
        vg_image_path, 
        "/data/yashowardhan/FineShot/test/results/vlm_vg_results.json", 
        vg_annotation_path,
        "vg", 
        "vlm"
    )
    
    
