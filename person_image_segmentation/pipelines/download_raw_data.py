import argparse
import os

from dotenv import load_dotenv

from person_image_segmentation.config import RAW_DATA_DIR, DATASET_LINK, KAGGLE_KEY, KAGGLE_USERNAME
from person_image_segmentation.utils.dataset_utils import download_dataset

# Load environment variables from a .env file
load_dotenv()
os.environ['KAGGLE_USERNAME'] = KAGGLE_KEY
os.environ['KAGGLE_KEY'] = KAGGLE_USERNAME


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))
    
    # Download the dataset
    download_dataset(
        dataset_link = DATASET_LINK,
        data_dir = RAW_DATA_DIR
    )