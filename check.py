import cv2
import supervision as sv
from rfdetr import RFDETRBase
from collections import Counter

model = RFDETRBase(
    pretrain_weights="runs/rfdetr_hardhat/checkpoint_best_ema.pth"
)

model.optimize_for_inference(batch_size=1)

# Get class names safely
if isinstance(model.class_names, dict):
    class_list = list(model.class_names.values())
else:
    class_list = model.class_names

cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = model.predict(rgb, threshold=0.3)

    print(f"\n========== Frame {frame_count} ==========")

    if len(detections) > 0:

        class_counter = Counter()

        for class_id, confidence, box in zip(
            detections.class_id,
            detections.confidence,
            detections.xyxy
        ):
            class_name = class_list[int(class_id)]
            class_counter[class_name] += 1

            print(
                f"Class: {class_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Box: {box}"
            )

        print("\nClass Count in This Frame:")
        for cls, count in class_counter.items():
            print(f"{cls}: {count}")

    else:
        print("No detections")

    # Prepare labels for drawing
    labels = [
        f"{class_list[int(class_id)]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated = box_annotator.annotate(frame, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    cv2.imshow("RF-DETR PPE", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
