import os
import json
from models.utils import compute_map, load_image_paths, load_annotations


refcoco_image_path = "/data/yashowardhan/FineShot/test/refcoco_images"
refcoco_annotation_path = "/data/yashowardhan/FineShot/data/refcoco data/refcoco_updated.json"

vg_image_path = "/data/yashowardhan/FineShot/test/vg_images"
vg_annotation_path = "/data/yashowardhan/FineShot/data/visual genome data/vg_subset.json"

model_configs = {
    "owlv2":{
        "confidence_threshold": [0.1, 0.5, 0.9],
        "index": [0, 100]
    },
    "yoloworld":{
        "confidence_threshold": [0.1, 0.5, 0.9],
        "index": [0, 100]
    },
    "yolo_clip":{
        "confidence_threshold": [0.1, 0.5, 0.9],
        "similarity_threshold": [0.25, 0.5, 0.75],
        "index": [0, 100]
    },
    "yolo_vlm":{
        "confidence_threshold": [0.1, 0.5, 0.9],
        "index": [45, 55]
    },
    # "yolo_clip_vlm":{
    #     "confidence_threshold": [0.1, 0.5, 0.9],
    #     "similarity_threshold": [0.25, 0.5, 0.75],
    #     "index": [45, 55]
    # }
}

def run_eval(image_path, annotation_path, dataset="refcoco"):
    image_paths = load_image_paths(image_path)
    annotations = load_annotations(annotation_path)
    iou_threshold = [0.5, 0.75, 0.9]
    map_results = []
    for model, config in model_configs.items():
        print(f"Running evaluation for {model}")
        
        if "similarity_threshold" in config:
            for ct in config["confidence_threshold"]:
                file_ct = str(ct).replace(".", "")
                for st in config["similarity_threshold"]:
                    file_st = str(st).replace(".", "")
                    result_file_path = f"/data/yashowardhan/FineShot/test/results/{model}_{file_ct}_{file_st}_{dataset}_results.json"   
                    with open(result_file_path, "r") as f:
                        results = json.load(f)
                    
                    for iou in iou_threshold:
                        counter = 1
                        all_boxes = []
                        all_annotations = []
                        for index in range(config["index"][0], config["index"][1]):
                            if dataset == "refcoco":
                                annotation = annotations[str(index+1)]["boxes"]   
                            else:
                                annotation = annotations[index+1]["boxes"]
                            
                            detections = results[str(counter)]["boxes"]
                            labels = results[str(counter)]["labels"]
                            
                            # Filter bounding boxes
                            filtered_boxes = []
                            for i in range(len(detections)):
                                if labels[i] != None:
                                    filtered_boxes.append(detections[i])
                            
                            all_boxes.append(filtered_boxes)
                            all_annotations.append(annotation)
                            counter += 1
                            
                        mAP = compute_map(all_boxes, all_annotations, iou)        
                        map_result = dict()
                        map_result["model"] = model
                        map_result["confidence_threshold"] = ct
                        map_result["similarity_threshold"] = st
                        map_result["iou_threshold"] = iou
                        map_result["mAP"] = mAP
                        map_results.append(map_result)
        else:
            for ct in config["confidence_threshold"]:
                file_ct = str(ct).replace(".", "")
                result_file_path = f"/data/yashowardhan/FineShot/test/results/{model}_{file_ct}_{dataset}_results.json" 
                
                with open(result_file_path, "r") as f:
                    results = json.load(f)
                
                for iou in iou_threshold:
                    counter = 1
                    all_boxes = []
                    all_annotations = []
                    for index in range(config["index"][0], config["index"][1]):
                        if dataset == "refcoco":
                            annotation = annotations[str(index+1)]["boxes"]   
                        else:
                            annotation = annotations[index+1]["boxes"]
                        
                        detections = results[str(counter)]["boxes"]
                        labels = results[str(counter)]["labels"]
                        
                        # Filter bounding boxes
                        filtered_boxes = []
                        for i in range(len(detections)):
                            if labels[i] != None:
                                filtered_boxes.append(detections[i])
                        
                        all_boxes.append(filtered_boxes)
                        all_annotations.append(annotation)
                        counter += 1
                        
                    mAP = compute_map(all_boxes, all_annotations, iou)
                    
                    map_result = dict()
                    map_result["model"] = model
                    map_result["confidence_threshold"] = ct
                    map_result["iou_threshold"] = iou
                    map_result["mAP"] = mAP
                    map_results.append(map_result)       

    return map_results


refcoco_map = run_eval(refcoco_image_path, refcoco_annotation_path, "refcoco")
with open("/data/yashowardhan/FineShot/test/results/refcoco_map_results.json", "w") as f:
    json.dump(refcoco_map, f, indent=4)
print("RefCOCO mAP results saved")

vg_map = run_eval(vg_image_path, vg_annotation_path, "vg")
with open("/data/yashowardhan/FineShot/test/results/vg_map_results.json", "w") as f:
    json.dump(vg_map, f, indent=4)
print("VG mAP results saved")