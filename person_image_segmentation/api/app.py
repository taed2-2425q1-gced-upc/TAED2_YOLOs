"""This module defines a FastAPI application for image segmentation using the YOLO model."""
import os
import time
import shutil
from pathlib import Path
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from codecarbon import EmissionsTracker # pylint: disable=E0401
from contextlib import asynccontextmanager
import asyncio

from person_image_segmentation.api.schema import (
    PredictionResponse,
    ErrorResponse,
    RootResponse,
    PredictionAndEnergyResponse
)
from person_image_segmentation.utils.api_utils import predict_mask_function

# Load Kaggle credentials
load_dotenv()

def clean_old_images():
    """
    Cleans up old image files from the static directory.

    Files older than 10 minutes are deleted to free up storage space.
    """
    now = time.time()
    time_ago = now - 600  # Every 10 mins

    for file in Path(str(REPO_PATH) + "/static").iterdir():
        if file.name != "favicon.ico" and file.is_file():
            # Check the last modification of the file
            if file.stat().st_mtime < time_ago:
                file.unlink()
                print(f"File {file.name} deleted!")

async def schedule_cleaning_task():
    """
    Schedules a periodic cleaning task to delete old files.

    The task runs every 60 seconds.
    """
    while True:
        clean_old_images()
        await asyncio.sleep(60)  # Run every 60 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Starts a background thread for the periodic cleaning task.
    """
    cleaning_task = asyncio.create_task(schedule_cleaning_task())
    yield

    cleaning_task.cancel()
    await cleaning_task

app = FastAPI(title="YOLOs image segmentation inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO template
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = Path(os.getenv('PATH_TO_BEST_WEIGHTS', 'yolov8m-seg.pt'))
BEST_WEIGHTS_FULL_PATH = str(REPO_PATH / BEST_WEIGHTS)

model = YOLO(BEST_WEIGHTS_FULL_PATH)

VALID_TOKEN = str(Path(os.getenv('VALID_TOKEN')))

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

@app.get("/", response_model=RootResponse)
def read_root():
    """
    Root endpoint providing a welcome message.

    Returns:
        RootResponse: A welcome message indicating the API is for YOLO predictions.
    """
    return RootResponse(message="API para hacer predicciones con YOLO")

@app.post("/predict/", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_mask(file: UploadFile = File(...), token: str = Depends(verify_token)): #pylint: disable = W0613
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
    # Save the file temporarily
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

        # Call the predict_mask_function to perform the prediction
        response = predict_mask_function(img_path, Path(REPO_PATH / "static"), img, model)

        # Delete temporary file
        os.remove(img_path)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Make sure to delete the temporary file
        if os.path.exists(img_path):
            os.remove(img_path)

@app.post("/predict_with_emissions/",
          response_model=PredictionAndEnergyResponse,
          responses={400: {"model": ErrorResponse}}
          )
async def predict_mask_with_emissions(
    file: UploadFile = File(...), token: str = Depends(verify_token)): #pylint: disable = W0613
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
    # Save the file temporarily
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

        # Predict and monitor emissions
        emissions_stats = {}
        with EmissionsTracker(
            output_dir=str(REPO_PATH / "static"), output_file="emissions_inference_api.csv"
            ) as _:
            response = predict_mask_function(img_path, Path(REPO_PATH / "static"), img, model)

        # Read the emissions file and return the results
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
        # Make sure to delete the temporary file
        if os.path.exists(img_path):
            os.remove(img_path)
