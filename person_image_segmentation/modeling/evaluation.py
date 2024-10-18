""" Evaluation script for the model """
import os
from pathlib import Path
import pandas as pd
import json
from PIL import Image
from dotenv import load_dotenv
from codecarbon import EmissionsTracker # pylint: disable=E0401
import mlflow
import numpy as np


# Load environment variables from a .env file
load_dotenv()

# Declare the base data path
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))

""" Function to compute mean Intersection over Union (mIoU) from the predictions """
def compute_miou(image_file_list: list[str], predictions_folder: Path) -> float:
    """
    Compute mean Intersection over Union (mIoU) for the provided list of image files
    and their corresponding predictions.

    ### Args:
        image_file_list (list[str]): List of image file paths.
        predictions_folder (Path): Path to the folder containing predicted masks.

    ### Returns:
        float: The computed mean IoU (mIoU).
    """
    # Initialize mean IoU
    total_iou = 0
    for image_file in image_file_list:
        try:
            file_name = image_file.split('/')[-1]
            true_mask = np.asarray(
                Image.open(
                    image_file.replace("processed", "interim/transformed")
                              .replace("images", "masks")
                              .replace("jpg", "png")
                )
            )
            pred_mask = np.asarray(Image.open(predictions_folder / file_name))

            gtb = true_mask > 0
            predb = pred_mask > 0

            overlap = gtb * predb
            union = gtb + predb
            iou = overlap.sum() / union.sum()

            total_iou += iou
        except Exception as e:
            raise RuntimeError(
                f"Could not compute mIoU for image in {file_name} because of {e}"
            ) from e
    total_iou /= len(image_file_list)
    return total_iou


if __name__ == "__main__":
    # Define predictions folder
    PREDS_PATH = REPO_PATH / "predictions"
    METRICS_PATH = REPO_PATH / "metrics"

    # Get testing image names
    folder_path = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(PREDS_PATH)
    file_names = [str(folder_path / file) for file in file_names if os.path.isfile(str(folder_path / file))]

    with EmissionsTracker(output_dir=str(METRICS_PATH), output_file="emissions_evaluation.csv") as tracker:

        
        mlflow.set_experiment("image-segmentation-yolo")

        with mlflow.start_run(run_name="yolov8-training-v0-codecarbon-evaluation"):
            mean_iou = compute_miou(file_names, PREDS_PATH)

            # Save the evaluation metrics to a dictionary to be reused later
            metrics_dict = {"mIoU": mean_iou}

            # Log the evaluation metrics to MLflow
            mlflow.log_metrics(metrics_dict)

            # Save the evaluation metrics to a JSON file
            with open(METRICS_PATH / "scores.json", "w") as scores_file:
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
