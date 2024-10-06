# General imports
import os
import yaml
import shutil
import subprocess
import random
import cv2

from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

from person_image_segmentation.utils.processing import copy_files, from_raw_masks_to_image_masks, from_image_masks_to_labels

# Load environment variables from a .env file
load_dotenv()

# Declare the base data path
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))

# Read the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_LINK = config['dataPipelines']['dataDownloading']['datasetLink']

# Combine the base path and subdirectories 
DATA_DIR = BASE_DATA_PATH / Path(config['dataPipelines']['dataDownloading']['dataDirectory']).relative_to('/')
SPLIT_DATA_DIR = BASE_DATA_PATH / Path(config['dataPipelines']['splitData']['dataDirectory']).relative_to('/')
TRANSFORM_DATA_DIR = BASE_DATA_PATH / Path(config['dataPipelines']['transformMasks']['dataDirectory']).relative_to('/')
LABELS_DATA_DIR = BASE_DATA_PATH / Path(config['dataPipelines']['createLabels']['dataDirectory']).relative_to('/')

# Declare split sizes
TRAIN_SIZE = config['dataPipelines']['splitData']['trainSize']
VAL_SIZE = config['dataPipelines']['splitData']['valSize']
TEST_SIZE = config['dataPipelines']['splitData']['testSize']

# Create data directory if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from a .env file and set up Kaggle credentials from environment variables
load_dotenv()
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')


# Function definition
def download_dataset(dataset_link: str, data_dir: Path) -> None:
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_link, path = data_dir, unzip = True)


def split_dataset(train_size: float, val_size: float, test_size: float, data_dir: Path, split_dir: Path) -> None:
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
    shutil.copytree(images_dir_val, images_dir_labels_train, dirs_exist_ok = True)
    shutil.copytree(images_dir_test, images_dir_labels_train, dirs_exist_ok = True)

if __name__ == "__main__":
    # First, download the dataset
    download_dataset(
        dataset_link = DATASET_LINK,
        data_dir = DATA_DIR
    )

    # Then split into train, val and test
    split_dataset(
        train_size = TRAIN_SIZE,
        val_size = VAL_SIZE,
        test_size = TEST_SIZE,
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
