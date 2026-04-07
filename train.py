from rfdetr import RFDETRBase

model = RFDETRBase(pretrained=True)

model.train(
    dataset_dir="/home/algosium/Downloads/main1ppe/coco_dataset",
    epochs=150,
    batch_size=8,
    grad_accum_steps=2,
    lr=2e-4,
    output_dir="runs/rfdetr_PPE-MAIN-1",
    resume=None
)
