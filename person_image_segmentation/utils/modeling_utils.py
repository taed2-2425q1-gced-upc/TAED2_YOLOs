import os
import random
import cv2
import shutil
import torch
import numpy as np

from PIL import Image
from pathlib import Path


def generate_predictions(test_filenames: list[str], predictions_folder: Path, model, max_predictions: int) -> None:
    # Create folder if it does not exist
    predictions_folder.mkdir(parents = True, exist_ok = True)

    # Predict mask for each image
    for idx, file in enumerate(test_filenames):
        if max_predictions and idx >= max_predictions:
            print("Early stop! The maximum number of predictions has been reached.")
            return
        
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