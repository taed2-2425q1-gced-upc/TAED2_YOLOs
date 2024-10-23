"""This module defines a FastAPI application for image segmentation using the YOLO model."""
import os
import time
import shutil
from pathlib import Path
from threading import Thread
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from codecarbon import EmissionsTracker # pylint: disable=E0401

from person_image_segmentation.api.schema import (
    PredictionResponse,
    ErrorResponse,
    RootResponse,
    PredictionAndEnergyResponse
)
from person_image_segmentation.utils.api_utils import predict_mask_function

# Load Kaggle credentials
load_dotenv()

app = FastAPI(title="YOLOs image segmentation inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de YOLO
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = Path(os.getenv('PATH_TO_BEST_WEIGHTS', 'yolov8m-seg.pt'))
BEST_WEIGHTS_FULL_PATH = str(REPO_PATH / BEST_WEIGHTS)

model = YOLO(BEST_WEIGHTS_FULL_PATH)

VALID_TOKEN = str(Path(os.getenv('VALID_TOKEN')))

# Configurar el esquema de seguridad para el token
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies if the provided token is valid.

    Args:
        credentials (HTTPAuthorizationCredentials): The authorization credentials containing 
        the token.

    Raises:
        HTTPException: If the token is invalid or not authorized.

    Returns:
        str: The valid token.
    """
    token = credentials.credentials
    if token != VALID_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token inválido o no autorizado",
        )
    return token

app.mount("/static", StaticFiles(
    directory = str(REPO_PATH) + "/static", html=True), name = "static"
    )

# Ruta para servir el favicon
@app.get("/favicon.ico", include_in_schema=True)
async def favicon():
    """
    Serves the favicon.ico file.

    Returns:
        FileResponse: The favicon file.
    """
    return FileResponse(str(REPO_PATH) + "/static/favicon.ico")

# Función para limpiar imágenes más antiguas de una hora
def clean_old_images():
    """
    Cleans up old image files from the static directory.

    Files older than 10 minutes are deleted to free up storage space.
    """
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
    """
    Schedules a periodic cleaning task to delete old files.

    The task runs every 60 seconds.
    """
    while True:
        clean_old_images()
        time.sleep(60)  # Ejecutar cada 60 segundos

# Configurar la ejecución de la tarea periódica en el evento de inicio
@app.on_event("startup")
async def startup_event():
    """
    Starts a background thread for the periodic cleaning task.
    """
    cleaning_thread = Thread(target=schedule_cleaning_task, daemon=True)
    cleaning_thread.start()

# Ruta base
@app.get("/", response_model=RootResponse)
def read_root():
    """
    Root endpoint providing a welcome message.

    Returns:
        RootResponse: A welcome message indicating the API is for YOLO predictions.
    """
    return RootResponse(message="API para hacer predicciones con YOLO")

# Ruta para hacer predicciones
@app.post("/predict/", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_mask(file: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Endpoint to make predictions for image segmentation.

    Args:
        file (UploadFile): The uploaded image file to be processed.
        token (str): The authorization token for verification.

    Returns:
        PredictionResponse: The response containing prediction details.

    Raises:
        HTTPException: If there is an error during processing.
    """
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

        # Llamar a la función predict_mask_function para realizar la predicción
        response = predict_mask_function(img_path, Path(REPO_PATH / "static"), img, model)

        # Eliminar archivo temporal
        os.remove(img_path)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Asegurarse de eliminar el archivo temporal
        if os.path.exists(img_path):
            os.remove(img_path)

# Nuevo endpoint para predicción y seguimiento de emisiones
@app.post("/predict_with_emissions/",
          response_model=PredictionAndEnergyResponse,
          responses={400: {"model": ErrorResponse}}
          )
async def predict_mask_with_emissions(
    file: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Endpoint to make predictions with energy consumption tracking.

    Args:
        file (UploadFile): The uploaded image file to be processed.
        token (str): The authorization token for verification.

    Returns:
        PredictionAndEnergyResponse: The response containing prediction details and energy 
        statistics.

    Raises:
        HTTPException: If there is an error during processing.
    """
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

        # Realizar predicción y seguimiento de emisiones
        emissions_stats = {}
        with EmissionsTracker(
            output_dir=str(REPO_PATH / "static"), output_file="emissions_inference_api.csv"
            ) as tracker:
            response = predict_mask_function(img_path, Path(REPO_PATH / "static"), img, model)

        # Leer el archivo de emisiones y devolver los resultados
        emissions_file = REPO_PATH / "static" / "emissions_inference_api.csv"
        if emissions_file.exists():
            emissions_data = pd.read_csv(emissions_file).to_dict(orient='records')
            if emissions_data:
                latest_emission = emissions_data[-1]
                emissions_stats = {
                    'emissions': latest_emission.get('emissions', None),
                    'duration': latest_emission.get('duration', None),
                    'cpu_power': latest_emission.get('cpu_power', None),
                    'gpu_power': latest_emission.get('gpu_power', None),
                    'ram_power': latest_emission.get('ram_power', None),
                    'energy_consumed': latest_emission.get('energy_consumed', None),
                }

        return PredictionAndEnergyResponse(
            prediction=response,
            energy_stats=emissions_stats,
            message="Prediction complete with energy tracking!"
            )
    finally:
        # Asegurarse de eliminar el archivo temporal
        if os.path.exists(img_path):
            os.remove(img_path)
