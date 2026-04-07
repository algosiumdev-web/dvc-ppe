import json
import os

splits = ["train", "valid", "test"]

for split in splits:
    json_path = f"coco_dataset/{split}/_annotations.coco.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    for cat in data["categories"]:
        if "supercategory" not in cat:
            cat["supercategory"] = "object"

    with open(json_path, "w") as f:
        json.dump(data, f)

print("✅ Supercategory added successfully!")
