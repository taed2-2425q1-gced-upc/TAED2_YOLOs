import numpy as np
import os
import cv2
import argparse

from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Declare the base data path
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))


def compute_mIoU(file_names: list[str], predictions_folder: Path) -> float:
    # Initialize mIoU
    mIoU = 0
    for idx, file in enumerate(file_names):
        try:
            file_name = file.split('/')[-1]
            true_mask = np.asarray(Image.open(file.replace("processed", "interim/transformed").replace("images", "masks").replace("jpg", "png")))
            pred_mask = np.asarray(Image.open(predictions_folder / file_name))

            gtb = (true_mask > 0)
            predb = (pred_mask > 0)

            overlap = gtb * predb
            union = gtb + predb
            IoU = overlap.sum() / union.sum()

            mIoU += IoU
        except Exception as e:
            raise Exception(f"Could not compute mIoU for image in {file_name} because of {e}")
    
    mIoU /= len(file_names)
    return mIoU



if __name__ == "__main__":
    # Define predictions folder
    PREDS_PATH = REPO_PATH / "predictions"

    # Get testing image names
    folder_path = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(PREDS_PATH)
    file_names = [str(folder_path / file) for file in file_names if os.path.isfile(str(folder_path / file))]

    mIoU = compute_mIoU(file_names, PREDS_PATH)

    print(f"Evaluation completed. Final mIoU is: {mIoU:.4f}")