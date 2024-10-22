import os
import torch
import numpy as np

from fastapi import HTTPException
from pathlib import Path
from PIL import Image
from datetime import datetime
from ultralytics import YOLO


from person_image_segmentation.api.schema import (
    PredictionResponse,
)

# Función para la predicción de la máscara
def predict_mask_function(img_path: str, output_dir: Path, img: Image.Image, model: YOLO):
    try:
        # Realizar predicción con YOLO
        results = model(img_path)
        result = results[0]

        # Verificar si existen máscaras en la predicción
        if not hasattr(result, 'masks') or result.masks is None:
            raise HTTPException(status_code=400, detail="No masks found in the prediction.")
            
        # Procesar la máscara predicha
        im = np.array(img)
        H, W = im.shape[0], im.shape[1]
        tmp_mask = result.masks.data
        tmp_mask, _ = torch.max(tmp_mask, dim=0)
        pred_mask = Image.fromarray(tmp_mask.cpu().numpy()).convert('P')
        pred_mask = pred_mask.resize((W, H))
        pred_mask = np.array(pred_mask)

        # Binarizar la máscara
        (width, height) = pred_mask.shape
        for y in range(height):
            for x in range(width):
                if pred_mask[x][y] > 0:
                    pred_mask[x][y] = 255

        # Generar el timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Convertir la máscara a imagen y guardarla
        im_to_save = Image.fromarray(pred_mask)
        output_filename = f"pred_{timestamp}_{os.path.basename(img_path)}"
        im_to_save.save(output_dir / output_filename)

        return PredictionResponse(filename=output_filename, message="Prediction complete!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
