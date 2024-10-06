from pathlib import Path
import numpy as np
from PIL import Image
from deepchecks.vision import VisionData, BatchOutputFormat
from deepchecks.vision.suites import data_integrity, train_test_validation

# Define paths
PROCESSED_DATA_DIR = Path("/Users/mariarisques/TAED2_YOLOs/data/interim/transformed")
REPORTS_DIR = Path("/Users/mariarisques/TAED2_YOLOs/data")

# Paths for images and labels for training and validation
train_images_dir = PROCESSED_DATA_DIR / 'images' / 'train'
train_labels_dir = PROCESSED_DATA_DIR / 'masks' / 'train'
val_images_dir = PROCESSED_DATA_DIR / 'images' / 'val'
val_labels_dir = PROCESSED_DATA_DIR / 'masks' / 'val'
test_images_dir = PROCESSED_DATA_DIR / 'images' / 'test'
test_labels_dir = PROCESSED_DATA_DIR / 'masks' / 'test'

# Get lists of image and label files
train_images_paths = sorted(list(train_images_dir.glob('*.jpg')))  
train_labels_paths = sorted(list(train_labels_dir.glob('*.png')))  
test_images_paths = sorted(list(test_images_dir.glob('*.jpg')))
test_labels_paths = sorted(list(test_labels_dir.glob('*.png')))
val_images_paths = sorted(list(test_images_dir.glob('*.jpg')))
val_labels_paths = sorted(list(test_labels_dir.glob('*.png')))

# Function to create the data generator
def custom_generator(images_paths, labels_paths, batch_size=64, target_size=(256, 256)):
    min_length = min(len(images_paths), len(labels_paths))
    for i in range(0, min_length, batch_size):
        images_batch = []
        labels_batch = []

        # Load images and masks for the batch
        for j in range(i, min(i + batch_size, min_length)):
            # Open and resize image
            img = Image.open(images_paths[j]).resize(target_size)
            img = np.array(img, dtype=np.uint8) 

            # Open and resize mask to match target size
            mask = Image.open(labels_paths[j]).resize(target_size)
            mask = np.array(mask, dtype=np.uint8) 

            # Ensure mask is single-channel (class IDs) if necessary
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # Use one channel if mask is RGB or multi-channel

            # Add to batch
            images_batch.append(img)
            labels_batch.append(mask)

        # Convert lists to numpy arrays
        images_batch = np.array(images_batch)  # Shape: (N, H, W, C)
        labels_batch = np.array(labels_batch)  # Shape: (N, H, W)

        # Yield the batch using BatchOutputFormat
        yield BatchOutputFormat(images=images_batch, labels=labels_batch)


# Create VisionData for train and test sets using the custom generator
train_ds = VisionData(custom_generator(train_images_paths, train_labels_paths), task_type='semantic_segmentation', reshuffle_data=False)
test_ds = VisionData(custom_generator(test_images_paths, test_labels_paths), task_type='semantic_segmentation', reshuffle_data=False)
val_ds = VisionData(custom_generator(val_images_paths, val_labels_paths), task_type='semantic_segmentation', reshuffle_data=False)

# Create the validation suite for data integrity and train-test validation
custom_suite = data_integrity()
custom_suite.add(train_test_validation())

# Run checks for train vs val
result_train_val = custom_suite.run(train_ds, val_ds)
result_train_val.save_as_html(str(REPORTS_DIR / "deepchecks_train_val_validation.html"))



