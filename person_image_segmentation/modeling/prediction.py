# General imports
import os
import yaml
import shutil
import cv2
import torch
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
from codecarbon import EmissionsTracker

# Load environment variables from a .env file
load_dotenv()

# Declare the base data path
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = os.getenv('PATH_TO_BEST_WEIGHTS')
best_weights_fullpath = str(REPO_PATH / BEST_WEIGHTS) if BEST_WEIGHTS != "None" else "yolov8m-seg.pt"


def generate_predictions(test_filenames: list[str], predictions_folder: Path, model, max_predictions: int) -> None:
    # Create folder if it does not exist
    predictions_folder.mkdir(parents = True, exist_ok = True)

    # Predict mask for each image
    for idx, file in enumerate(test_filenames):
        if max_predictions and idx >= max_predictions:
            print("Early stop! The maximum number of predictions has been reached.")
            return

        if file.split('/')[-1] == '.DS_Store':
            continue
        
        # Process the image
        results = model(file)
        result = results[0]
        if hasattr(result, 'masks') and result.masks is not None:
            try:
                im = cv2.imread(file)
                H, W = im.shape[0], im.shape[1]
                tmp_mask = result.masks.data
                tmp_mask, _ = torch.max(tmp_mask, dim = 0)
                pred_mask = Image.fromarray(tmp_mask.cpu().numpy()).convert('P')
                pred_mask = pred_mask.resize((W, H))
                pred_mask = np.array(pred_mask)

                (width, height) = pred_mask.shape
                for y in range(height):
                    for x in range(width):
                        if pred_mask[x][y] > 0:
                            pred_mask[x][y] = 255
                
                im_to_save = Image.fromarray(pred_mask)
                im_to_save.save(predictions_folder / file.split('/')[-1])
                file_name = file.split('/')[-1]
                if idx % 50 == 0:
                    print(f"Already processed {idx} images")

            except:
                raise Exception(f"Error processing image {file}")


if __name__ == "__main__":
    # Argument parser for max number of predictions
    parser = argparse.ArgumentParser(description="Generate predictions for test images.")
    parser.add_argument('--max_predictions', type=int, default=10, help='Maximum number of predictions to generate')
    args = parser.parse_args()
    max_predictions = args.max_predictions

    # Load the YOLO model
    model = YOLO(best_weights_fullpath)

    # Load the test images
    test_folder = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(test_folder)
    file_names = [str(test_folder / file) for file in file_names if os.path.isfile(str(test_folder / file))]
    
    with EmissionsTracker(output_dir=str(REPO_PATH / "metrics"), output_file="emissions_inference.csv") as tracker:
        # Make predictions
        PREDS_PATH = REPO_PATH / "predictions"
        generate_predictions(
            test_filenames = file_names,
            predictions_folder = PREDS_PATH,
            model = model,
            max_predictions = max_predictions
        )
