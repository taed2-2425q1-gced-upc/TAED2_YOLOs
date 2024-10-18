from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
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

app = FastAPI(title="YOLOs image segmentation inference")

# Cargar el modelo de YOLO
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = Path(os.getenv('PATH_TO_BEST_WEIGHTS', 'yolov8m-seg.pt'))
best_weights_fullpath = str(REPO_PATH / BEST_WEIGHTS)

model = YOLO(best_weights_fullpath)

VALID_TOKEN = "YOLOs"

# Configurar el esquema de seguridad para el token
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifica si el token es válido.
    """
    token = credentials.credentials
    if token != VALID_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token inválido o no autorizado",
        )
    return token

# Ruta para servir el favicon
@app.get("/favicon.ico", include_in_schema=True)
async def favicon():
    return FileResponse(str(REPO_PATH) + "/static/favicon.ico")  # Ruta al archivo favicon.ico

# Ruta base
@app.get("/", response_model=RootResponse)
def read_root():
    return RootResponse(message="API para hacer predicciones con YOLO")

# Ruta para hacer predicciones
@app.post("/predict/", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_mask(file: UploadFile = File(...), token: str = Depends(verify_token)):
    # Guardar el archivo temporalmente
    img_path = f"temp_{file.filename}"
    try:
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(img_path)
        if img.format != 'JPEG':
            img = img.convert('RGB')
            jpg_path = f"temp_{os.path.splitext(file.filename)[0]}.jpg"
            img.save(jpg_path, 'JPEG')
            os.remove(img_path)
            img_path = jpg_path

        # Realizar predicción con YOLO
        results = model(img_path)
        result = results[0]

        # Verificar si existen máscaras en la predicción
        if not hasattr(result, 'masks') or result.masks is None:
            # No se encontraron máscaras, devolver error
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
        pred_mask[pred_mask > 0] = 255

        # Convertir la máscara a imagen y devolverla
        im_to_save = Image.fromarray(pred_mask)
        im_to_save.save(f"pred_{file.filename}")

        # Eliminar archivo temporal
        os.remove(img_path)

        return PredictionResponse(filename=f"pred_{file.filename}", message="Prediction complete!")
    finally:
        # Asegurarse de eliminar el archivo temporal
        if os.path.exists(img_path):
            os.remove(img_path)


