!pip install datasets

import datasets
import json
import collections
from collections import defaultdict
from datasets import load_dataset

refcoco_dataset = load_dataset("lmms-lab/RefCOCO", split="val", streaming=True)

image_dict = defaultdict(list)

for i, item in enumerate(refcoco_dataset):
    if i >= 1000:
        break 

    file_name = item["file_name"]  
    bbox = item["bbox"]  # Format: [x, y, width, height]
    answers = item["answer"]  

    # Convert bbox to [x1, y1, x2, y2]
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height

    for phrase in answers:
        image_dict[file_name].append({
            "coordinates": [x1, y1, x2, y2],
            "phrase": phrase
        })

output_data = [{"image_id": image_id, "regions": regions} for image_id, regions in image_dict.items()]


# Save to JSON file
with open("refcoco_subset.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("JSON file saved: refcoco_subset.json")
