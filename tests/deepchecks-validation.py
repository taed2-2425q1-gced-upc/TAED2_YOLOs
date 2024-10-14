from pathlib import Path
import numpy as np
import os
from PIL import Image
from deepchecks.vision import VisionData, BatchOutputFormat
from deepchecks.vision.suites import data_integrity, train_test_validation
from dotenv import load_dotenv

def load_environment_vars():
    # Load Kaggle credentials
    load_dotenv()
    data_dir = Path(os.getenv('PATH_TO_DATA_FOLDER')) / "interim" / "transformed"
    repo_dir = Path(os.getenv('PATH_TO_REPO')) / "reports"
    return data_dir, repo_dir

def get_paths(data_dir):
    train_images_dir = data_dir / 'images' / 'train'
    train_labels_dir = data_dir / 'masks' / 'train'
    val_images_dir = data_dir / 'images' / 'val'
    val_labels_dir = data_dir / 'masks' / 'val'
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir

def list_files(images_dir, labels_dir):
    image_paths = sorted(list(images_dir.glob('*.jpg')))
    label_paths = sorted(list(labels_dir.glob('*.png')))
    return image_paths, label_paths

def custom_generator(images_paths, labels_paths, batch_size=64, target_size=(256, 256)):
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
    return VisionData(generator, task_type=task_type, reshuffle_data=False)

def run_checks(train_ds, val_ds, reports_dir):
    suite = data_integrity()
    suite.add(train_test_validation())
    result = suite.run(train_ds, val_ds)
    result.save_as_html(str(reports_dir / "deepchecks_train_val_validation.html"))

def main():
    data_dir, reports_dir = load_environment_vars()
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = get_paths(data_dir)
    train_images_paths, train_labels_paths = list_files(train_images_dir, train_labels_dir)
    val_images_paths, val_labels_paths = list_files(val_images_dir, val_labels_dir)
    train_ds = create_vision_data(custom_generator(train_images_paths, train_labels_paths), 'semantic_segmentation')
    val_ds = create_vision_data(custom_generator(val_images_paths, val_labels_paths), 'semantic_segmentation')
    run_checks(train_ds, val_ds, reports_dir)

if __name__ == "__main__":
    main()





