import os
import json
import glob
import cv2
import yaml

def convert_split(split_name, images_dir, labels_dir, output_json, categories):
    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    image_files = glob.glob(os.path.join(images_dir, "*.*"))

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        file_name = os.path.basename(img_path)

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })

        label_path = os.path.join(
            labels_dir,
            os.path.splitext(file_name)[0] + ".txt"
        )

        if os.path.exists(label_path):

            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, x, y, bw, bh = map(float, parts[:5])



                    x_min = (x - bw / 2) * w
                    y_min = (y - bh / 2) * h
                    width = bw * w
                    height = bh * h

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_format, f)


# Load class names from data.yaml
with open("data.yaml") as f:
    data = yaml.safe_load(f)

names = data["names"]
categories = [{"id": i, "name": name} for i, name in enumerate(names)]

os.makedirs("coco_dataset/annotations", exist_ok=True)
os.makedirs("coco_dataset/train", exist_ok=True)
os.makedirs("coco_dataset/val", exist_ok=True)
os.makedirs("coco_dataset/test", exist_ok=True)

# Convert each split
convert_split("train", "train/images", "train/labels",
              "coco_dataset/annotations/instances_train.json", categories)

convert_split("val", "valid/images", "valid/labels",
              "coco_dataset/annotations/instances_val.json", categories)

convert_split("test", "test/images", "test/labels",
              "coco_dataset/annotations/instances_test.json", categories)

print("✅ Conversion completed successfully!")
