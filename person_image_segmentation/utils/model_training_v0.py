import subprocess
import sys

# Install necessary packages (uncomment if required)
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "ultralytics", "mlflow", "dagshub", "codecarbon"])

import os
import mlflow
import dagshub
from codecarbon import EmissionsTracker
from ultralytics import YOLO

# Disable W&B logging
os.environ["WANDB_MODE"] = "offline"

# Set the tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/nachoogriis/TAED2_YOLOs.mlflow")

# Initialize DagsHub integration
dagshub.init(
   repo_name="TAED2_YOLOs",
   repo_owner="nachoogriis",
)

# Set experiment name in MLflow
mlflow.set_experiment("test-experiments")

# Enable MLflow autologging
mlflow.autolog()

# Load the YOLO model
model = YOLO('yolov8m-seg.pt')

# Path to the config file in the Kaggle environment
config_file_path = "/kaggle/input/yolo-training-data/config_yolos.yaml"

# Use CodeCarbon's EmissionsTracker as a context manager
with EmissionsTracker(gpu_ids=[]) as tracker:
   # Start a new MLflow run to track the experiment
   with mlflow.start_run(run_name="YoloV8-Training-Test-CodeCarbon"):

       # Training the model
       results = model.train(data=config_file_path, epochs=1, imgsz=640, name="Yolo Weights")

       print("Entrenamiento completado y experimentos registrados en MLflow.")
