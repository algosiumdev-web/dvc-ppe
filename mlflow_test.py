from ultralytics import YOLO
import mlflow
import os

# ✅ Setup MLflow
# os.makedirs("/home/algosium/mlruns", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ppe_yolo_training22")


# ✅ Callback for epoch-wise logging
def on_epoch_end(trainer):
    metrics = trainer.metrics

    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                
                # ✅ FIX: clean metric name
                clean_key = k.replace("(", "").replace(")", "").replace(" ", "_")

                mlflow.log_metric(clean_key, float(v), step=trainer.epoch)

    print(f"📊 Logged epoch {trainer.epoch}")



# Load model
model = YOLO("yolo11n.pt")


# ✅ ADD CALLBACK HERE
model.add_callback("on_fit_epoch_end", on_epoch_end)

with mlflow.start_run(run_name="test55"):

    # ✅ Log params (safe now)
    params = {
        "epochs": 25,
        "batch": 16,
        "imgsz": 768,
        "optimizer": "auto",
        "cos_lr": True,
        "mosaic": 0.6,
        "device": 0
    }
    mlflow.log_params(params)

    print("✅ MLflow run started")

    # ✅ Train
    results = model.train(
        data="data.yaml",
        epochs=25,
        batch=16,
        imgsz=768,
        device=0,
        name="test55",
        project="runs",
        cos_lr=True,
        mosaic=0.6,
        optimizer="auto"
    )

    # ✅ Log metrics (after training)
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            if isinstance(v, (int, float)):
                
                clean_key = k.replace("(", "").replace(")", "").replace(" ", "_")
                
                mlflow.log_metric(clean_key, float(v))

    # ✅ Log artifacts
    save_dir = str(results.save_dir)
    mlflow.log_artifacts(save_dir)

    # ✅ Log best model
    best_model_path = os.path.join(save_dir, "weights/best.pt")
    if os.path.exists(best_model_path):
        mlflow.log_artifact(best_model_path)