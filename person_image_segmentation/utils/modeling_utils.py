"""
Module for generating predictions using a machine learning model.

This module provides functions to generate predictions for test images using a specified model
and save the resulting masks in the specified directory.
"""

from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image


def generate_predictions(test_filenames: list[str], predictions_folder: Path,
                         model, max_predictions: int) -> None:
    """
    Generate predictions for a list of test images using a given model.

    Args:
        test_filenames (list[str]): List of file paths for the test images.
        predictions_folder (Path): Path to the directory where the prediction masks will be saved.
        model: The model used for generating predictions.
        max_predictions (int): Maximum number of predictions to generate. If set to 0, all images 
                                will be processed.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while processing an image.
    """
    # Create folder if it does not exist
    predictions_folder.mkdir(parents=True, exist_ok=True)

    def process_and_save_image(file, results):
        """Processes an image, generates a mask, and saves it."""
        im = cv2.imread(file)  # pylint: disable=no-member
        h, w = im.shape[:2]
        tmp_mask = results[0].masks.data
        tmp_mask, _ = torch.max(tmp_mask, dim=0)
        pred_mask = Image.fromarray(tmp_mask.cpu().numpy()).convert('P')
        pred_mask = pred_mask.resize((w, h))
        pred_mask = np.array(pred_mask)

        pred_mask[pred_mask > 0] = 255
        im_to_save = Image.fromarray(pred_mask)
        im_to_save.save(predictions_folder / Path(file).name)

    # Predict mask for each image
    for idx, file in enumerate(test_filenames):
        if max_predictions and idx >= max_predictions:
            print("Early stop! The maximum number of predictions has been reached.")
            return

        try:
            results = model(file)
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                process_and_save_image(file, results)
                if idx % 50 == 0:
                    print(f"Already processed {idx} images")

        except Exception as exc:
            raise RuntimeError(f"Error processing image {file}") from exc


def compute_miou(image_file_list: list[str], predictions_folder: Path) -> float:
    """
    Compute mean Intersection over Union (mIoU) for the provided list of image files
    and their corresponding predictions.

    Args:
        image_file_list (list[str]): List of image file paths.
        predictions_folder (Path): Path to the folder containing predicted masks.

    Returns:
        float: The computed mean IoU (mIoU).
    """
    total_iou = 0
    for image_file in image_file_list:
        try:
            file_name = Path(image_file).name
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

            overlap = np.logical_and(gtb, predb)
            union = np.logical_or(gtb, predb)
            iou = overlap.sum() / union.sum()
            total_iou += iou
        except Exception as e:
            raise RuntimeError(
                f"Could not compute mIoU for image {file_name} because of {e}"
            ) from e
    return total_iou / len(image_file_list)
