"""
Simple training script for YOLOv8 model.

This script configures and trains a YOLOv8 model for image segmentation using 
MLflow for experiment tracking, DagsHub for version control, and CodeCarbon 
for tracking carbon emissions. It supports both CPU and GPU training.
"""

import os
import mlflow
import dagshub
import torch

from codecarbon import EmissionsTracker
from ultralytics import YOLO

from person_image_segmentation.config import REPO_PATH

# Print GPU availability for debugging
print("Is CUDA available?:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "No GPU Found")

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
config_file_path = REPO_PATH / "models/configs/config_yolos_sample_train.yaml"
cfg_file_path_hyps = REPO_PATH / "models/configs/config_hyps.yaml"

# Use CodeCarbon's EmissionsTracker as a context manager
with EmissionsTracker(gpu_ids=[0]) as tracker:
    # Start a new MLflow run to track the experiment
    with mlflow.start_run(run_name="YoloV8-training-v0-Hyps"):
        # Training the model
        if not torch.cuda.is_available():
            results = model.train(
                data=config_file_path,
                epochs=1,
                imgsz=640,
                cfg = cfg_file_path_hyps,
                name="Sample_Train__DVC_Pipeline"
                )
        else:
            results = model.train(
                data=config_file_path,
                epochs=1,
                imgsz=640,
                cfg = cfg_file_path_hyps,
                name="Sample_Train__DVC_Pipeline",
                device = 0
                )

        print("Training completed and experiments recorded in MLflow.")
