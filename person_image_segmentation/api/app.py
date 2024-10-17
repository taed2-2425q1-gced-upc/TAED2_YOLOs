from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from PIL import Image
import os
import torch
import numpy as np
import shutil
from ultralytics import YOLO
from dotenv import load_dotenv

from person_image_segmentation.api.schema import (
    PredictionResponse,
    ErrorResponse,
    RootResponse,
)

# Load Kaggle credentials
load_dotenv()

app = FastAPI(title = "YOLOs image segmentation inference")

# Cargar el modelo de YOLO
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = Path(os.getenv('PATH_TO_BEST_WEIGHTS', 'yolov8m-seg.pt'))
best_weights_fullpath = str(REPO_PATH / BEST_WEIGHTS)

model = YOLO(best_weights_fullpath)

# Ruta base
@app.get("/", response_model=RootResponse)
def read_root():
    return RootResponse(message="API para hacer predicciones con YOLO")

# Ruta para hacer predicciones
@app.post("/predict/", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_mask(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = Image.open(img_path)
    print(img.format)
    if img.format != 'JPEG':
        img = img.convert('RGB')  # Convertir a RGB
        jpg_path = f"temp_{os.path.splitext(file.filename)[0]}.jpg"  # Nueva ruta con extensión JPG
        img.save(jpg_path, 'JPEG')  # Guardar como JPG
        os.remove(img_path)
        img_path = jpg_path  # Actualizar la ruta de la imagen a la nueva

    
    # Leer la imagen
    img = Image.open(img_path)
    
    # Realizar predicción con YOLO
    results = model(img_path)
    result = results[0]
    
    if hasattr(result, 'masks') and result.masks is not None:
        try:
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

            # Convertir la máscara a imagen y devolverla
            im_to_save = Image.fromarray(pred_mask)
            im_to_save.save(f"pred_{file.filename}")

            # Eliminar archivo temporal
            os.remove(img_path)

            return PredictionResponse(filename=f"pred_{file.filename}", message="Prediction complete!")
        except Exception as e:
            return ErrorResponse(error=str(e))

    else:
        return ErrorResponse(error="No masks found in the prediction.")