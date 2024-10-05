import subprocess
import sys

# Install necessary packages (uncomment if required)
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "ultralytics", "mlflow", "dagshub"])

import os
import mlflow
import dagshub
from ultralytics import YOLO

# Disable W&B logging
os.environ["WANDB_MODE"] = "offline"

# Set the tracking URI for DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Initialize DagsHub integration
dagshub.init(
    repo_name="TAED2_YOLOs",
    repo_owner="nachoogriis",
)

# Set experiment name in MLflow
mlflow.set_experiment("test-experiment-yolo-v0")

# Enable MLflow autologging
mlflow.autolog()

# Load the YOLO model
model = YOLO('yolov8m-seg.pt')

# Path to the config file in the Kaggle environment
config_file_path = "/kaggle/input/yolo-training-data/config_yolos.yaml"

# Start a new MLflow run to track the experiment
with mlflow.start_run(run_name="YoloV8-Training"):
    
    # Tuning the model
    # model.tune(data=config_file_path, epochs=1, imgsz=640, name="Hyps Tuning Second Version", iterations=30)
    
    # Training the model
    results = model.train(data=config_file_path, epochs=1, imgsz=640, name="Yolo Weights")

    print("Entrenamiento completado y experimentos registrados en MLflow.")




