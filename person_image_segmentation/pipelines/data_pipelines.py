""" This module contains the data pipelines and functions for the project """
# General imports
import os
import shutil
from pathlib import Path
import random
import argparse

from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

from person_image_segmentation.utils.processing import (
    copy_files,
    from_raw_masks_to_image_masks,
    from_image_masks_to_labels
)

# Import constants and configurations
from person_image_segmentation.config import (
    DATA_DIR,
    DATASET_LINK,
    SPLIT_DATA_DIR,
    TRANSFORM_DATA_DIR,
    LABELS_DATA_DIR,
    TRAIN_SIZE,
    VAL_SIZE,
    KAGGLE_KEY,
    KAGGLE_USERNAME
)

# Load environment variables from a .env file
load_dotenv()

# Load environment variables from a .env file and set up Kaggle credentials
os.environ['KAGGLE_USERNAME'] =KAGGLE_KEY
os.environ['KAGGLE_KEY'] =KAGGLE_USERNAME


# Function definition
def download_dataset(dataset_link: str, data_dir: Path) -> None:
    """ 
        Downloads the dataset from Kaggle 
        ### Args
        - `dataset_link` -> link to the dataset in kaggle relative to the username
        - `data_dir` -> path to the directory where the dataset will be downloaded
    """
    # Create data directory if it does not exist
    DATA_DIR.mkdir(parents = True, exist_ok = True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_link, path = data_dir, unzip = True)


def split_dataset(
    train_size: float,
    val_size: float,
    data_dir: Path,
    split_dir: Path
) -> None:
    """
    Splits the dataset into train, val and test sets
    ### Args
    - `train_size` -> size of the train set
    - `val_size` -> size of the val set
    - `test_size` -> size of the test set
    - `data_dir` -> path to the directory where the dataset is stored
    - `split_dir` -> path to the directory where the splits will be stored
    """
    # Define directories
    images_dir = data_dir / 'dataset_person-yolos/data/images'
    masks_dir = data_dir / 'dataset_person-yolos/data/masks'
    images_dir_train = split_dir / 'images/train'
    masks_dir_train = split_dir / 'masks/train'
    images_dir_val = split_dir / 'images/val'
    masks_dir_val = split_dir / 'masks/val'
    images_dir_test = split_dir / 'images/test'
    masks_dir_test = split_dir / 'masks/test'

    # Get the list of samples and shuffle them
    samples = os.listdir(images_dir)
    random.shuffle(samples)

    # Calculate split indices
    num_samples = len(samples)
    train_end = int(train_size * num_samples)
    val_end = train_end + int(val_size * num_samples)

    # Split samples
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Create necessary directories
    split_dir.mkdir(parents = True, exist_ok = True)
    images_dir_train.mkdir(parents = True, exist_ok = True)
    masks_dir_train.mkdir(parents = True, exist_ok = True)
    images_dir_val.mkdir(parents = True, exist_ok = True)
    masks_dir_val.mkdir(parents = True, exist_ok = True)
    images_dir_test.mkdir(parents = True, exist_ok = True)
    masks_dir_test.mkdir(parents = True, exist_ok = True)

    # Copy files to respective directories
    copy_files(train_samples, images_dir, masks_dir, images_dir_train, masks_dir_train)
    copy_files(val_samples, images_dir, masks_dir, images_dir_val, masks_dir_val)
    copy_files(test_samples, images_dir, masks_dir, images_dir_test, masks_dir_test)


def transform_masks(split_dir: Path, transform_dir: Path) -> None:
    """
    Transforms the masks to a format suitable for training the model
    ### Args
    - `split_dir` -> path to the directory where the splits are stored
    - `transform_dir` -> path to the directory where the transformed masks will be stored
    """
    # Define directories
    input_dir_train = split_dir / 'masks/train'
    output_dir_train = transform_dir / 'masks/train'
    input_dir_val = split_dir / 'masks/val'
    output_dir_val = transform_dir / 'masks/val'
    input_dir_test = split_dir / 'masks/test'
    output_dir_test = transform_dir / 'masks/test'
    images_dir_train = split_dir / 'images/train'
    images_dir_val = split_dir / 'images/val'
    images_dir_test = split_dir / 'images/test'

    output_dir_train.mkdir(parents = True, exist_ok = True)
    output_dir_val.mkdir(parents = True, exist_ok = True)
    output_dir_test.mkdir(parents = True, exist_ok = True)

    from_raw_masks_to_image_masks(
        input_dirs = [input_dir_train, input_dir_val, input_dir_test],
        output_dirs = [output_dir_train, output_dir_val, output_dir_test]
    )

    images_dir_trans_train = transform_dir / 'images/train'
    images_dir_trans_val = transform_dir / 'images/val'
    images_dir_trans_test = transform_dir / 'images/test'

    shutil.copytree(images_dir_train, images_dir_trans_train, dirs_exist_ok = True)
    shutil.copytree(images_dir_val, images_dir_trans_val, dirs_exist_ok = True)
    shutil.copytree(images_dir_test, images_dir_trans_test, dirs_exist_ok = True)


def generate_labels(transform_dir: Path, labels_dir: Path, split_dir: Path) -> None:
    """
    Generates labels for the transformed masks
    ### Args
    - `transform_dir` -> path to the directory where the transformed masks are stored
    - `labels_dir` -> path to the directory where the labels will be stored
    - `split_dir` -> path to the directory where the splits are stored    
    """
    # Define directories
    input_dir_train = transform_dir / 'masks/train'
    output_dir_train = labels_dir / 'labels/train'
    input_dir_val = transform_dir / 'masks/val'
    output_dir_val = labels_dir / 'labels/val'
    input_dir_test = transform_dir / 'masks/test'
    output_dir_test = labels_dir / 'labels/test'
    images_dir_train = split_dir / 'images/train'
    images_dir_val = split_dir / 'images/val'
    images_dir_test = split_dir / 'images/test'

    output_dir_train.mkdir(parents = True, exist_ok = True)
    output_dir_val.mkdir(parents = True, exist_ok = True)
    output_dir_test.mkdir(parents = True, exist_ok = True)

    from_image_masks_to_labels(
        input_dirs = [input_dir_train, input_dir_val, input_dir_test],
        output_dirs = [output_dir_train, output_dir_val, output_dir_test]
    )

    images_dir_labels_train = labels_dir / 'images/train'
    images_dir_labels_val = labels_dir / 'images/val'
    images_dir_labels_test = labels_dir / 'images/test'

    shutil.copytree(images_dir_train, images_dir_labels_train, dirs_exist_ok = True)
    shutil.copytree(images_dir_val, images_dir_labels_val, dirs_exist_ok = True)
    shutil.copytree(images_dir_test, images_dir_labels_test, dirs_exist_ok = True)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()
    if args.test:
        DATA_DIR = Path(str(DATA_DIR).replace('data', 'test_data'))
        SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
        TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))
        LABELS_DATA_DIR = Path(str(LABELS_DATA_DIR).replace('data', 'test_data'))

    # First, download the dataset
    download_dataset(
        dataset_link = DATASET_LINK,
        data_dir = DATA_DIR
    )

    # Then split into train, val and test
    split_dataset(
        train_size = TRAIN_SIZE,
        val_size = VAL_SIZE,
        data_dir = DATA_DIR,
        split_dir = SPLIT_DATA_DIR
    )

    # Then transform masks
    transform_masks(
        split_dir = SPLIT_DATA_DIR,
        transform_dir = TRANSFORM_DATA_DIR
    )

    # Then generate labels
    generate_labels(
        transform_dir = TRANSFORM_DATA_DIR,
        labels_dir = LABELS_DATA_DIR,
        split_dir = SPLIT_DATA_DIR
    )
