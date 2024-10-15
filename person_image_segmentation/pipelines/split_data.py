import argparse

from pathlib import Path

from person_image_segmentation.config import RAW_DATA_DIR, SPLIT_DATA_DIR, TRAIN_SIZE, VAL_SIZE, TEST_SIZE
from person_image_segmentation.utils.dataset_utils import split_dataset


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))
        SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
    
    # Split into train, val and test
    split_dataset(
        train_size = TRAIN_SIZE,
        val_size = VAL_SIZE,
        test_size = TEST_SIZE,
        data_dir = RAW_DATA_DIR,
        split_dir = SPLIT_DATA_DIR
    )