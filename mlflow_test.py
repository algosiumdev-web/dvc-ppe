from rfdetr import RFDETRBase
import mlflow
import os
import json

# =========================
# ✅ MLflow Setup
# =========================
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("sqlite:////home/algosium/mlflow_server/mlflow.db")
mlflow.set_experiment("ppe_rfdetr_experiment")

# =========================
# ✅ PATHS (IMPORTANT)
# =========================
output_dir = "runs/rfdetr_PPE-MAIN-1"

results_json_path = os.path.join(output_dir, "results.json")
log_file_path = os.path.join(output_dir, "log.txt")
plot_path = os.path.join(output_dir, "metrics_plot.png")

best_model_path = os.path.join(output_dir, "checkpoint_best_ema.pth")

# =========================
# ✅ START RUN
# =========================
with mlflow.start_run(run_name="rfdetr_ppe_main1"):

    print("✅ MLflow run started (OFFLINE LOGGING)")

    # =========================
    # ✅ TAGS (for comparison)
    # =========================
    mlflow.set_tags({
        "model": "rfdetr",
        "type": "detection",
        # "dataset": "ppe",
        "dataset": "ppe_v1",
        "dvc_remote": "myremote",
        "mode": "offline_logging"
    })

    # ✅ ADD THIS HERE
    mlflow.set_tag("data_version", "v1_PPE_MERGED_DATASET_V1.0")  



    # =========================
    # ✅ PARAMS
    # =========================
    mlflow.log_params({
        "epochs": 150,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "lr": 2e-4,
        "model": "RF-DETR Base"
    })

    # =========================
    # ✅ LOAD RESULTS.JSON
    # =========================
    data = None

    if os.path.exists(results_json_path):
        with open(results_json_path, "r") as f:
            data = json.load(f)
        print("✅ results.json loaded")
    else:
        print("❌ results.json NOT FOUND")

    # =========================
    # ✅ LOG FINAL METRICS
    # =========================
    if data:
        try:
            mlflow.log_metric("mAP", data["map"])
            mlflow.log_metric("precision", data["precision"])
            mlflow.log_metric("recall", data["recall"])
            mlflow.log_metric("f1_score", data["f1_score"])

            print("✅ Final metrics logged")
        except Exception as e:
            print("⚠️ Error logging final metrics:", e)

        # =========================
        # ✅ CLASS-WISE METRICS
        # =========================
        try:
            if "class_map" in data and "valid" in data["class_map"]:
                class_data = data["class_map"]["valid"]

                for cls in class_data:
                    name = cls["class"].replace(" ", "_")

                    mlflow.log_metric(f"{name}_map50", cls["map@50"])
                    mlflow.log_metric(f"{name}_precision", cls["precision"])
                    mlflow.log_metric(f"{name}_recall", cls["recall"])

                print("✅ Class-wise metrics logged")
        except Exception as e:
            print("⚠️ Error logging class-wise metrics:", e)

    # =========================
    # ✅ EPOCH-WISE METRICS (log.txt)
    # =========================
    if os.path.exists(log_file_path):
        print("✅ Logging epoch-wise metrics...")

        with open(log_file_path, "r") as f:
            for step, line in enumerate(f):
                try:
                    log = json.loads(line)

                    if "train_loss" in log:
                        mlflow.log_metric("train_loss", log["train_loss"], step=step)

                    if "test_loss" in log:
                        mlflow.log_metric("val_loss", log["test_loss"], step=step)

                    if "ema_test_loss" in log:
                        mlflow.log_metric("ema_val_loss", log["ema_test_loss"], step=step)

                except:
                    continue

        print("✅ Epoch-wise metrics logged")
    else:
        print("❌ log.txt NOT FOUND")

    # =========================
    # ✅ LOG FULL OUTPUT FOLDER
    # =========================
    if os.path.exists(output_dir):
        mlflow.log_artifacts(output_dir)
        print("✅ Full artifacts logged")

    # =========================
    # ✅ LOG PLOT IMAGE
    # =========================
    if os.path.exists(plot_path):
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print("✅ Plot logged")

    # =========================
    # ✅ LOG MODEL
    # =========================
    if os.path.exists(best_model_path):
        mlflow.log_artifact(best_model_path, artifact_path="model")
        print("✅ Model logged")

    print("🚀 ALL DATA SUCCESSFULLY UPLOADED TO MLFLOW")