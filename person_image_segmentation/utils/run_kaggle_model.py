import os
import subprocess
import requests
import json

from dotenv import load_dotenv
from pathlib import Path


def load_kaggle_credentials():
    """Load Kaggle credentials from the environment variables."""
    load_dotenv()

    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')

    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise ValueError("Kaggle credentials are not set properly in the .env file.")
    
    return KAGGLE_USERNAME, KAGGLE_KEY


def set_kaggle_env_vars(KAGGLE_USERNAME, KAGGLE_KEY):
    """Set Kaggle environment variables for API."""
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY


def create_kernel_metadata(KAGGLE_USERNAME, code_file_path):
    """Create kernel metadata for Kaggle."""
    dataset_slug = f'{KAGGLE_USERNAME.lower()}/yolo-training-data'  # Ensure the username and slug are lowercase
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
            dataset_slug  # Reference to the uploaded dataset
        ]
    }
    return kernel_metadata


def save_kernel_metadata(kernel_metadata, kernel_metadata_path='kernel-metadata.json'):
    """Save kernel metadata to a JSON file."""
    with open(kernel_metadata_path, 'w') as f:
        json.dump(kernel_metadata, f, indent=4)


def push_kernel_to_kaggle():
    """Push the kernel to Kaggle."""
    print("Pushing kernel to Kaggle...")
    push_result = subprocess.run(['kaggle', 'kernels', 'push', '-p', '.'])
    
    if push_result.returncode != 0:
        raise RuntimeError("Failed to push the kernel to Kaggle. Please check the output for errors.")
    
    return push_result


def execute_kernel(KAGGLE_USERNAME, KAGGLE_KEY):
    """Execute the kernel manually (optional step)."""
    kernel_slug = f'{KAGGLE_USERNAME.lower()}/entrenamiento-yolo'
    response = requests.post(
        f'https://www.kaggle.com/api/v1/kernels/{kernel_slug}/status',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {KAGGLE_KEY}'
        }
    )
    return response


def main():
    """Main function to execute the entire process."""
    # Load credentials
    KAGGLE_USERNAME, KAGGLE_KEY = load_kaggle_credentials()

    # Set environment variables
    set_kaggle_env_vars(KAGGLE_USERNAME, KAGGLE_KEY)

    # Paths for local files
    code_file_path = Path(os.getenv('PATH_TO_REPO')) / 'person_image_segmentation/modeling/train_v0.py'

    # Create and save kernel metadata
    kernel_metadata = create_kernel_metadata(KAGGLE_USERNAME, code_file_path)
    save_kernel_metadata(kernel_metadata)

    # Push the kernel to Kaggle
    push_kernel_to_kaggle()

    # Execute the kernel (optional)
    execute_kernel(KAGGLE_USERNAME, KAGGLE_KEY)


if __name__ == '__main__':
    main()
