"""
Module for generating predictions for the api endpoints.

This module provides functions to generate predictions for test images using a specified model
and save the resulting masks in the specified directory.
"""

import os
from datetime import datetime
from pathlib import Path
from http import HTTPStatus

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image
from ultralytics import YOLO

from person_image_segmentation.api.schema import PredictionResponse

# Function to predict the mask
def predict_mask_function(
        img_path: str,
        output_dir: Path,
        img: Image.Image,
        model: YOLO
    ) -> PredictionResponse:
    """
    Predicts the mask for the given image using the specified YOLO model.

    Args:
        img_path (str): Path to the input image.
        output_dir (Path): Directory where the output mask will be saved.
        img (Image.Image): Image object of the input image.
        model (YOLO): The YOLO model used for prediction.

    Returns:
        PredictionResponse: Response object containing the filename and message.

    Raises:
        HTTPException: If no masks are found or if an error occurs during processing.
    """
    try:
        # Perform prediction with YOLO
        results = model(img_path)
        result = results[0]

        # Check if masks exist in the prediction
        if not hasattr(result, 'masks') or result.masks is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="No masks found in the prediction."
                )

        # Process the predicted mask
        img_array = np.array(img)
        height, width = img_array.shape[:2]  # Changed H, W to height, width for snake_case
        tmp_mask = result.masks.data
        tmp_mask, _ = torch.max(tmp_mask, dim=0)
        pred_mask = Image.fromarray(tmp_mask.cpu().numpy()).convert('P')
        pred_mask = pred_mask.resize((width, height))
        pred_mask = np.array(pred_mask)

        # Binarize the mask
        pred_mask[pred_mask > 0] = 255  # More efficient way to binarize the mask

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Convert the mask to an image and save it
        im_to_save = Image.fromarray(pred_mask)
        output_filename = f"pred_{timestamp}_{os.path.basename(img_path)}"
        im_to_save.save(output_dir / output_filename)

        return PredictionResponse(filename=output_filename, message="Prediction complete!")

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)) from e
