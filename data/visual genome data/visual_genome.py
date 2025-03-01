!pip install datasets

import json
import datasets

from datasets import load_dataset
vg_dataset = load_dataset("visual_genome", "region_descriptions_v1.2.0", split ="train", streaming = True)

output_data = []
for i, item in enumerate(vg_dataset):
    if i >= 1000:
        break  # Stop after 1000 images
    
    image_id = item["image_id"]
    regions = [
        {
            "coordinates": [region["x"], region["y"], region["x"] + region["width"], region["y"] + region["height"]],
            "phrase": region["phrase"]
        }
        for region in item["regions"]
    ]
    output_data.append({"image_id": image_id, "regions": regions})
    
# Saving
with open("vg_subset.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("JSON file saved: vg_subset.json")