import os
import torch
import time
import numpy as np
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
from threading import Thread

from person_image_segmentation.api.schema import (
    PredictionResponse,
    ErrorResponse,
    RootResponse,
)

# Load Kaggle credentials
load_dotenv()

app = FastAPI(title="YOLOs image segmentation inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's domain if needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de YOLO
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = Path(os.getenv('PATH_TO_BEST_WEIGHTS', 'yolov8m-seg.pt'))
best_weights_fullpath = str(REPO_PATH / BEST_WEIGHTS)

model = YOLO(best_weights_fullpath)

VALID_TOKEN = str(Path(os.getenv('VALID_TOKEN')))
print(VALID_TOKEN)

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

app.mount("/static", StaticFiles(directory = str(REPO_PATH) + "/static", html=True), name = "static")

# Ruta para servir el favicon
@app.get("/favicon.ico", include_in_schema=True)
async def favicon():
    return FileResponse(str(REPO_PATH) + "/static/favicon.ico")  # Ruta al archivo favicon.ico


from threading import Thread

# Función para limpiar imágenes más antiguas de una hora
def clean_old_images():
    now = time.time()
    time_ago = now - 600  # Cada 10 mins

    for file in Path(str(REPO_PATH) + "/static").iterdir():
        if file.name != "favicon.ico" and file.is_file():
            # Verificar la última modificación del archivo
            if file.stat().st_mtime < time_ago:
                try:
                    file.unlink()  # Eliminar el archivo
                    print(f"File {file.name} deleted!")
                except Exception as e:
                    print(f"Failed to delete {file.name}: {str(e)}")

# Función para ejecutar la limpieza periódicamente
def schedule_cleaning_task():
    while True:
        clean_old_images()
        time.sleep(60)  # Ejecutar cada 60 segundos 

# Configurar la ejecución de la tarea periódica en el evento de inicio
@app.on_event("startup")
async def startup_event():
    cleaning_thread = Thread(target=schedule_cleaning_task, daemon=True)
    cleaning_thread.start()

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

                # Generar el timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Convertir la máscara a imagen y devolverla
                im_to_save = Image.fromarray(pred_mask)
                im_to_save.save(str(REPO_PATH) + "/static/"+f"pred_{timestamp}_{file.filename}")

                # Eliminar archivo temporal
                os.remove(img_path)

                return PredictionResponse(filename=f"pred_{timestamp}_{file.filename}", message="Prediction complete!")
            except Exception as e:
                return ErrorResponse(error=str(e))

        else:
            return ErrorResponse(error="No masks found in the prediction.")
          
    finally:
        # Asegurarse de eliminar el archivo temporal
        if os.path.exists(img_path):
            os.remove(img_path)


