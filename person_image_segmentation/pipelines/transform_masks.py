import argparse

from dotenv import load_dotenv

from person_image_segmentation.config import SPLIT_DATA_DIR, TRANSFORM_DATA_DIR
from person_image_segmentation.utils.dataset_utils import transform_masks


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
        TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))
    
    # Transform masks
    transform_masks(
        split_dir = SPLIT_DATA_DIR,
        transform_dir = TRANSFORM_DATA_DIR
    )