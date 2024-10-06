import os
import subprocess
import requests
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
code_file_path = os.PATH.join(os.getenv('PATH_TO_REPO') ,'/person_image_segmentation/utils/model_training_v0.py')

# Define the dataset slug
dataset_slug = f'{KAGGLE_USERNAME.lower()}/yolo-training-data'  # Ensure the username and slug are lowercase

# Create kernel metadata for Kaggle
kernel_metadata = {
    'id': f'{KAGGLE_USERNAME.lower()}/entrenamiento-yolo',  # Kernel ID in lowercase
    'title': 'YoloV8-training-v0-Hyps',
    'code_file': code_file_path,
    'language': 'python',
    'kernel_type': 'script',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
    'dataset_sources': [
        dataset_slug  # Reference to the uploaded dataset
    ]
}

# Save kernel metadata
kernel_metadata_path = 'kernel-metadata.json'
with open(kernel_metadata_path, 'w') as f:
    json.dump(kernel_metadata, f, indent=4)

# Push the kernel to Kaggle
print("Pushing kernel to Kaggle...")
push_result = subprocess.run(['kaggle', 'kernels', 'push', '-p', '.'])

# Check if the kernel was pushed successfully
if push_result.returncode != 0:
    raise RuntimeError("Failed to push the kernel to Kaggle. Please check the output for errors.")

# Execute the kernel (optional step, kernel will execute automatically after being pushed)
kernel_slug = f'{KAGGLE_USERNAME.lower()}/entrenamiento-yolo'
response = requests.post(
    f'https://www.kaggle.com/api/v1/kernels/{kernel_slug}/status',
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {KAGGLE_KEY}'
    }
)
