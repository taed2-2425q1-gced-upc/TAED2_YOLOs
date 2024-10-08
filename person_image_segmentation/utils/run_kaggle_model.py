""" Module to run a model in Kaggle"""
import os
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import requests


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
code_file_path = Path(os.getenv('PATH_TO_REPO')) / 'person_image_segmentation/modeling/train_v0.py'

# Define the dataset slug
DATASET_SLUG = f'{KAGGLE_USERNAME.lower()}/yolo-training-data'  # Ensure lowercase

# Create kernel metadata for Kaggle
kernel_metadata = {
    'id': f'{KAGGLE_USERNAME.lower()}/entrenamiento-yolo',  # Kernel ID in lowercase
    'title': 'YoloV8-training-v0-Hyps',
    'code_file': str(code_file_path),
    'language': 'python',
    'kernel_type': 'script',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
    'dataset_sources': [
        DATASET_SLUG  # Reference to the uploaded dataset
    ]
}

# Save kernel metadata
KERNEL_METADATA_PATH = 'kernel-metadata.json'
with open(KERNEL_METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(kernel_metadata, f, indent=4)

# Push the kernel to Kaggle
print("Pushing kernel to Kaggle...")
push_result = subprocess.run(['kaggle', 'kernels', 'push', '-p', '.'], check=False)

# Check if the kernel was pushed successfully
if push_result.returncode != 0:
    raise RuntimeError("Failed to push the kernel to Kaggle. Please check the output for errors.")

# Execute the kernel (optional step, kernel will execute automatically after being pushed)
KERNEL_SLUG = f'{KAGGLE_USERNAME.lower()}/entrenamiento-yolo'
response = requests.post(
    f'https://www.kaggle.com/api/v1/kernels/{KERNEL_SLUG}/status',
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {KAGGLE_KEY}'
    },
    timeout=30
)
