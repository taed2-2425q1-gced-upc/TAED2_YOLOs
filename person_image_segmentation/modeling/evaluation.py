""" Evaluation script for the model """
import os
from pathlib import Path
import json
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from codecarbon import EmissionsTracker # pylint: disable=E0401
import mlflow
import numpy as np

from person_image_segmentation.utils.evaluation_utils import compute_miou

# Load environment variables from a .env file
load_dotenv()

# Declare the base data path
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))


if __name__ == "__main__":
    # Define predictions folder
    PREDS_PATH = REPO_PATH / "predictions"
    METRICS_PATH = REPO_PATH / "metrics"
    OUTPUT_FILE = "emissions_evaluation.csv"

    # Get testing image names
    folder_path = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(PREDS_PATH)
    file_names = [
        str(folder_path / file) for file in file_names
        if os.path.isfile(str(folder_path / file))
        ]

    with EmissionsTracker(output_dir=str(METRICS_PATH), output_file=OUTPUT_FILE) as tracker:

        mlflow.set_experiment("image-segmentation-yolo")

        with mlflow.start_run(run_name="yolov8-training-v0-codecarbon-evaluation"):
            mean_iou = compute_miou(file_names, PREDS_PATH)

            # Save the evaluation metrics to a dictionary to be reused later
            metrics_dict = {"mIoU": mean_iou}

            # Log the evaluation metrics to MLflow
            mlflow.log_metrics(metrics_dict)

            # Save the evaluation metrics to a JSON file
            with open(METRICS_PATH / "scores.json", "w", encoding='utf-8') as scores_file:
                json.dump(
                    metrics_dict,
                    scores_file,
                    indent=4,
                )
            emissions = pd.read_csv(METRICS_PATH / "emissions_evaluation.csv")
            emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
            emissions_params = emissions.iloc[-1, 13:].to_dict()
            mlflow.log_params(emissions_params)
            mlflow.log_metrics(emissions_metrics)

    print(f"Evaluation completed. Final mIoU is: {mean_iou:.4f}")
