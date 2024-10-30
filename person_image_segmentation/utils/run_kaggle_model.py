""" Module to run a model in Kaggle"""
import os
import subprocess
import json
from pathlib import Path
import requests

from person_image_segmentation.config import REPO_PATH, KAGGLE_USERNAME, KAGGLE_KEY

def load_kaggle_credentials():
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise ValueError("Kaggle credentials are not set properly in the .env file.")

    return KAGGLE_USERNAME, KAGGLE_KEY


def create_kernel_metadata(kaggle_username, code_file_path):
    """Create kernel metadata for Kaggle."""
    dataset_slug = f'{kaggle_username.lower()}/yolo-training-data'
    kernel_metadata = {
        'id': f'{kaggle_username.lower()}/entrenamiento-yolo', 
        'title': 'yolov8-training-v0-hyps',
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
    with open(kernel_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(kernel_metadata, f, indent=4)


def push_kernel_to_kaggle():
    """Push the kernel to Kaggle."""
    print("Pushing kernel to Kaggle...")
    push_result = subprocess.run(['kaggle', 'kernels', 'push', '-p', '.'], check = True)

    if push_result.returncode != 0:
        raise RuntimeError("Failed to push the kernel to Kaggle.")

    return push_result


def execute_kernel(kaggle_username, kaggle_key):
    """Execute the kernel manually (optional step)."""
    kernel_slug = f'{kaggle_username.lower()}/entrenamiento-yolo'
    response = requests.post(
        f'https://www.kaggle.com/api/v1/kernels/{kernel_slug}/status',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {kaggle_key}'
        },
        timeout=3600
    )
    return response


def main():
    """Main function to execute the entire process."""
    # Load credentials
    kaggle_username, kaggle_key = load_kaggle_credentials()

    # Paths for local files
    code_file_path = (
        REPO_PATH / 'person_image_segmentation/modeling/train_v0.py'
    )

    # Create and save kernel metadata
    kernel_metadata = create_kernel_metadata(kaggle_username, code_file_path)
    save_kernel_metadata(kernel_metadata)

    # Push the kernel to Kaggle
    push_kernel_to_kaggle()

    # Execute the kernel (optional)
    execute_kernel(kaggle_username, kaggle_key)


if __name__ == '__main__':
    main()
