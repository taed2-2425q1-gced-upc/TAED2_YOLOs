"""
Utility functions for dataset management.

This module provides functions to download datasets, split them into training, validation, 
and test sets, transform masks, generate labels, and manage data folder structure for machine 
learning workflows.
"""

import os
import random
import shutil

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

from person_image_segmentation.utils.processing import (
    copy_files,
    from_raw_masks_to_image_masks,
    from_image_masks_to_labels
)

def download_dataset(dataset_link: str, data_dir: Path) -> None:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_link (str): Kaggle dataset link.
        data_dir (Path): Path to the directory where the dataset will be downloaded.

    Returns:
        None
    """
    # Create data directory if it does not exist
    data_dir.mkdir(parents = True, exist_ok = True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_link, path = data_dir, unzip = True)


def split_dataset(train_size: float, val_size: float,
                  data_dir: Path, split_dir: Path) -> None:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.
        data_dir (Path): Path to the directory containing the original dataset.
        split_dir (Path): Path to the directory where the split dataset will be saved.

    Returns:
        None
    """
    # Define directories
    data_paths = {
        "images": data_dir / 'dataset_person-yolos/data/images',
        "masks": data_dir / 'dataset_person-yolos/data/masks'
    }
    split_paths = {
        "train": {
            "images": split_dir / 'images/train',
            "masks": split_dir / 'masks/train'
        },
        "val": {
            "images": split_dir / 'images/val',
            "masks": split_dir / 'masks/val'
        },
        "test": {
            "images": split_dir / 'images/test',
            "masks": split_dir / 'masks/test'
        }
    }

    # Get the list of samples and shuffle them
    samples = os.listdir(data_paths["images"])
    random.shuffle(samples)

    # Calculate split indices and split samples
    train_end = int(train_size * len(samples))
    val_end = train_end + int(val_size * len(samples))
    sample_splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }

    # Create necessary directories
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in split_paths.values():
        for path in split.values():
            path.mkdir(parents=True, exist_ok=True)

    # Copy files to respective directories
    for split, split_samples in sample_splits.items():
        copy_files(split_samples, data_paths["images"], data_paths["masks"],
                   split_paths[split]["images"], split_paths[split]["masks"])


def transform_masks(split_dir: Path, transform_dir: Path) -> None:
    """
    Transform mask images from raw format to image masks format.

    Args:
        split_dir (Path): Path to the directory containing the split dataset 
                          (training, validation, and test).
        transform_dir (Path): Path to the directory where the transformed masks will be saved.

    Returns:
        None
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
    Generate labels from transformed masks.

    Args:
        transform_dir (Path): Path to the directory containing the transformed masks.
        labels_dir (Path): Path to the directory where the generated labels will be saved.
        split_dir (Path): Path to the directory containing the original split dataset.

    Returns:
        None
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


def complete_data_folder(config_names: list[str], src_folder: Path, dst_folder: Path) -> None:
    """
    Copy configuration files from the source folder to the destination folder.

    Args:
        config_names (list[str]): List of configuration file names to copy.
        src_folder (Path): Path to the source folder containing the configuration files.
        dst_folder (Path): Path to the destination folder where the files will be copied.

    Returns:
        None
    """
    for config_name in config_names:
        shutil.copy(src_folder / config_name, dst_folder / config_name)
        