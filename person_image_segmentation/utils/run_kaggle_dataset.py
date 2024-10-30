"""Run the dataset creation and upload to Kaggle."""
import os
import subprocess
import json

from person_image_segmentation.config import DATA_DIR, KAGGLE_USERNAME

def set_paths():
    """Set paths for local files."""
    data_directory_path = DATA_DIR
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
    # Retrieve the paths
    data_directory_path, dataset_metadata_path = set_paths()

    # Create and upload dataset
    create_and_upload_dataset(KAGGLE_USERNAME, data_directory_path, dataset_metadata_path)

# Call the main function
if __name__ == "__main__":
    main()
