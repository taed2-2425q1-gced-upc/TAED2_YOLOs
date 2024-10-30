"""
Module for running data integrity and validation checks on image segmentation datasets.

This module provides functions to:
- Load environment variables to set up data and report directories.
- Prepare paths for training and validation datasets, including images and labels.
- Create a custom data generator for batching images and labels with optional resizing.
- Set up VisionData objects for use with deepchecks for semantic segmentation tasks.
- Run deepchecks suites to validate the integrity of the training and validation datasets and 
  generate an HTML report with the results.

Main Features:
- Environment setup using .env file for paths configuration.
- Custom data generator that supports image and mask resizing.
- Data integrity and train-test validation checks using deepchecks.
- Report generation in HTML format for easy visualization of results.

Usage:
    Run the script directly to perform the checks and generate the report:
    ```bash
    python <script_name>.py
    ```
"""


from pathlib import Path
import os
import numpy as np
from PIL import Image
from deepchecks.vision import VisionData, BatchOutputFormat
from deepchecks.vision.suites import data_integrity, train_test_validation

from person_image_segmentation.config import DATA_DIR, REPO_PATH

def load_environment_vars():
    """
    Load environment variables and set the data and repository directories.

    Returns:
        tuple: A tuple containing:
            - data_dir (Path): Path to the data directory.
            - repo_dir (Path): Path to the repository reports directory.
    """

    data_dir = DATA_DIR / "interim" / "transformed"
    repo_dir = REPO_PATH / "reports"
    return data_dir, repo_dir

def get_paths(data_dir):
    """
    Get the paths for training and validation images and labels.

    Args:
        data_dir (Path): The base data directory.

    Returns:
        tuple: A tuple containing:
            - train_images_dir (Path): Path to the training images directory.
            - train_labels_dir (Path): Path to the training labels directory.
            - val_images_dir (Path): Path to the validation images directory.
            - val_labels_dir (Path): Path to the validation labels directory.
    """

    train_images_dir = data_dir / 'images' / 'train'
    train_labels_dir = data_dir / 'masks' / 'train'
    val_images_dir = data_dir / 'images' / 'val'
    val_labels_dir = data_dir / 'masks' / 'val'
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir

def list_files(images_dir, labels_dir):
    """
    List the image and label files in the specified directories.

    Args:
        images_dir (Path): The directory containing the image files.
        labels_dir (Path): The directory containing the label files.

    Returns:
        tuple: A tuple containing:
            - image_paths (list[Path]): List of image file paths.
            - label_paths (list[Path]): List of label file paths.
    """

    image_paths = sorted(list(images_dir.glob('*.jpg')))
    label_paths = sorted(list(labels_dir.glob('*.png')))
    return image_paths, label_paths

def custom_generator(images_paths, labels_paths, batch_size=64, target_size=(256, 256)):
    """
    Custom generator that yields batches of images and labels.

    Args:
        images_paths (list[Path]): List of image file paths.
        labels_paths (list[Path]): List of label file paths.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        target_size (tuple, optional): Size to which the images and labels should be resized. 
        Defaults to (256, 256).

    Yields:
        BatchOutputFormat: A batch containing images and labels in the required format.
    """

    min_length = min(len(images_paths), len(labels_paths))
    for i in range(0, min_length, batch_size):
        images_batch = []
        labels_batch = []
        for j in range(i, min(i + batch_size, min_length)):
            img = Image.open(images_paths[j]).resize(target_size)
            img = np.array(img, dtype=np.uint8)
            mask = Image.open(labels_paths[j]).resize(target_size)
            mask = np.array(mask, dtype=np.uint8)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]
            images_batch.append(img)
            labels_batch.append(mask)
        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        yield BatchOutputFormat(images=images_batch, labels=labels_batch)

def create_vision_data(generator, task_type):
    """
    Create a VisionData object for the given generator and task type.

    Args:
        generator (generator): A generator yielding batches of images and labels.
        task_type (str): The type of task (e.g., 'semantic_segmentation').

    Returns:
        VisionData: An object representing the data for deepchecks.
    """

    return VisionData(generator, task_type=task_type, reshuffle_data=False)

def run_checks(train_ds, val_ds, reports_dir):
    """
    Run deepchecks data integrity and train-test validation checks.

    Args:
        train_ds (VisionData): Training dataset for deepchecks.
        val_ds (VisionData): Validation dataset for deepchecks.
        reports_dir (Path): Directory to save the reports.

    Returns:
        None
    """

    suite = data_integrity()
    suite.add(train_test_validation())
    result = suite.run(train_ds, val_ds)
    result.save_as_html(str(reports_dir / "deepchecks_train_val_validation.html"))

def main():
    """
    Main function to run the data integrity and validation checks.

    It loads the environment variables, prepares the data paths, creates the datasets,
    and runs the deepchecks checks on the training and validation datasets.

    Returns:
        None
    """

    data_dir, reports_dir = load_environment_vars()
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = get_paths(data_dir)
    train_images_paths, train_labels_paths = list_files(train_images_dir, train_labels_dir)
    val_images_paths, val_labels_paths = list_files(val_images_dir, val_labels_dir)
    train_ds = create_vision_data(
        custom_generator(train_images_paths, train_labels_paths), 'semantic_segmentation'
        )
    val_ds = create_vision_data(
        custom_generator(val_images_paths, val_labels_paths), 'semantic_segmentation'
        )
    run_checks(train_ds, val_ds, reports_dir)

if __name__ == "__main__":
    main()
