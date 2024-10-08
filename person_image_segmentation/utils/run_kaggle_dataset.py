"""Run the dataset creation and upload to Kaggle."""
import os
import subprocess
import json
from dotenv import load_dotenv

# Load Kaggle credentials
load_dotenv()

KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')

# Validate credentials
if not KAGGLE_USERNAME or not KAGGLE_KEY:
    raise ValueError("Kaggle credentials are not set properly in the .env file.")

# Set environment variables for Kaggle API
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

# Paths for local files
code_file_path = os.path.join(
    os.getenv('PATH_TO_REPO'),
    'person_image_segmentation/modeling/train_v0.py'
)

data_directory_path = os.getenv('PATH_TO_DATA_FOLDER')
dataset_metadata_path = os.path.join(data_directory_path, 'dataset-metadata.json')

# Define the dataset slug
DATASET_SLUG = f'{KAGGLE_USERNAME.lower()}/yolo-training-data'  # Use lowercase for consistency

# 1. Create and Upload the Dataset to Kaggle
print("\nStep 1: Creating and Uploading Dataset")

# Create dataset metadata for Kaggle
dataset_metadata = {
    "title": "YOLO Training Data",  # A descriptive title for your dataset
    "id": DATASET_SLUG,
    "licenses": [
        {
            "name": "CC0-1.0"
        }
    ]
}

# Save dataset metadata in the data directory
with open(dataset_metadata_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_metadata, f, indent=4)

# Debug: Check if the file was created successfully
print("Current directory:", os.getcwd())
print("Checking if dataset-metadata.json exists:", os.path.exists(dataset_metadata_path))

# Change to the directory containing dataset-metadata.json
os.chdir(data_directory_path)

# Upload data as a Kaggle dataset (including subfolders)
print("Uploading dataset to Kaggle...")
upload_result = subprocess.run(['kaggle', 'datasets', 'create', '-p', '.', '--dir-mode', 'zip'], check=False)

# Check if the dataset was uploaded successfully
if upload_result.returncode != 0:
    raise RuntimeError(
    "Failed to upload the dataset to Kaggle. "
    "Please check the output above for errors."
)

print("Dataset uploaded successfully.")
