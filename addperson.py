from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm


IMAGE_DIR = "valid/images"
LABEL_DIR = "valid/labels"

PERSON_CLASS_ID = 2
CONF_THRESHOLD = 0.4


model = YOLO("yolo11n.pt")  # pretrained COCO model

# Filter images once
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png"))
]


for img_name in tqdm(image_files, desc="Annotating images", unit="img"):

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Load existing annotations
    existing_labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            existing_labels = f.readlines()

    # Run YOLOv11m
    results = model(img, conf=CONF_THRESHOLD, verbose=False)

    new_labels = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0]

                # Convert to YOLO format
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                new_labels.append(
                    f"{PERSON_CLASS_ID} {xc} {yc} {bw} {bh}\n"
                )

    # Save merged labels
    with open(label_path, "w") as f:
        if existing_labels:
            # ensure last line ends with newline
            if not existing_labels[-1].endswith("\n"):
                existing_labels[-1] += "\n"
            f.writelines(existing_labels)

        f.writelines(new_labels)


print("✅ Person annotations added successfully!")
