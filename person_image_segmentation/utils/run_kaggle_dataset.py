"""Run the dataset creation and upload to Kaggle."""
import os
import subprocess
import json
from dotenv import load_dotenv

# Load Kaggle credentials
load_dotenv()

def validate_kaggle_credentials():
    """Validate Kaggle credentials from environment variables."""
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        raise ValueError("Kaggle credentials are not set properly in the .env file.")

    # Set environment variables for Kaggle API
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

def set_paths():
    """Set paths for local files."""
    data_directory_path = os.getenv('PATH_TO_DATA_FOLDER')
    dataset_metadata_path = os.path.join(data_directory_path, 'dataset-metadata.json')
    return data_directory_path, dataset_metadata_path

def create_and_upload_dataset(kaggle_username, data_directory_path, dataset_metadata_path):
    """Create and upload the dataset to Kaggle."""
    # Define the dataset slug
    dataset_slug = f'{kaggle_username.lower()}/yolo-training-data'  # Use lowercase for consistency

    # Step 1: Creating and Uploading Dataset
    print("\nStep 1: Creating and Uploading Dataset")

    # Create dataset metadata for Kaggle
    dataset_metadata = {
        "title": "YOLO Training Data",  # A descriptive title for your dataset
        "id": dataset_slug,
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
    upload_result = subprocess.run(
        ['kaggle', 'datasets', 'create', '-p', '.', '--dir-mode', 'zip'],
        check=True
    )

    # Check if the dataset was uploaded successfully
    if upload_result.returncode != 0:
        raise RuntimeError("Failed to upload the dataset to Kaggle.")

    print("Dataset uploaded successfully.")

# Main function to execute the process
def main():
    """Main function to run the dataset creation and upload process."""
    validate_kaggle_credentials()

    # Retrieve the paths
    data_directory_path, dataset_metadata_path = set_paths()

    # Get Kaggle username for dataset creation
    kaggle_username = os.getenv('KAGGLE_USERNAME')

    # Create and upload dataset
    create_and_upload_dataset(kaggle_username, data_directory_path, dataset_metadata_path)

# Call the main function
if __name__ == "__main__":
    main()
