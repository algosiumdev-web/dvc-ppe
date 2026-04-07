import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRNano()

image = Image.open("/home/algosium/Downloads/CJ1322-Electric_Neon_Green_Sequin_Mirror_Vest_1__61199.jpg")
detections = model.predict(image, threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)