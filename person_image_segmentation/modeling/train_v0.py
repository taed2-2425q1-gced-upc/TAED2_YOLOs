"""
Training script for YOLOv8 model.

This script configures and trains a YOLOv8 model for image segmentation using 
MLflow for experiment tracking, DagsHub for version control, and CodeCarbon 
for tracking carbon emissions. It supports both CPU and GPU training.
"""

# pylint: disable = R0801
import subprocess
import sys
import os

# Install necessary packages (uncomment if required)
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "opencv-python", "ultralytics", "mlflow",
    "dagshub", "codecarbon"
])

import mlflow # pylint: disable=C0413
import dagshub # pylint: disable=C0413
from codecarbon import EmissionsTracker # pylint: disable=E0401 disable=C0413
from ultralytics import YOLO # pylint: disable=E0401 disable=C0413

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
mlflow.set_experiment("image-segmentation-yolo")

# Enable MLflow autologging
mlflow.autolog()

# Load the YOLO model
model = YOLO('yolov8m-seg.pt')

# Path to the config file in the Kaggle environment
CONFIG_FILE_PATH = "/kaggle/input/yolo-training-data/config_yolos.yaml"
CFG_FILE_PATH_HYPS = "/kaggle/input/yolo-training-data/config_hyps.yaml"

# Use CodeCarbon's EmissionsTracker as a context manager
with EmissionsTracker(gpu_ids=[]) as tracker:
    # Start a new MLflow run to track the experiment
    with mlflow.start_run(run_name="YoloV8-training-v0-Hyps"):

        # Training the model
        results = model.train(
            data=CONFIG_FILE_PATH,
            epochs=100,
            imgsz=640,
            cfg = CFG_FILE_PATH_HYPS,
            name="Yolo Weights")

        print("Training completed and experiments recorded in MLflow.")
