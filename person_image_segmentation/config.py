
""" Configuration file for the project """
from pathlib import Path

import os

from tqdm import tqdm
from dotenv import load_dotenv
from loguru import logger # pylint: disable=E0401

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv('PATH_TO_DATA_FOLDER'))
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
DATASET_LINK = "mariarisques/dataset-person-yolos"
CONFIG_PATH = REPO_PATH / 'person_image_segmentation/config.yaml'

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TRANSFORM_DATA_DIR = DATA_DIR / "interim" / "transformed"
SPLIT_DATA_DIR = DATA_DIR / "interim" / "splitted"
LABELS_DATA_DIR = DATA_DIR / "processed"

# Split sizes
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Kaggle credentials
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')

# DagsHub
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
DAGSHUB_REPO_OWNER = os.getenv('DAGSHUB_REPO_OWNER')

# MLFlow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Best model weights
PATH_TO_BEST_WEIGHTS = os.getenv('PATH_TO_BEST_WEIGHTS')

# Valid API token
VALID_TOKEN = os.getenv('VALID_TOKEN')

logger.remove(0)
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
