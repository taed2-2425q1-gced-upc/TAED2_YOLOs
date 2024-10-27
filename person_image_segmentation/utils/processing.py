""" Module with processing utils """
import shutil
import os
import cv2
from PIL import Image

def copy_files(sample_list, src_images_dir, src_masks_dir, dest_images_dir, dest_masks_dir):
    """ Copies files from source to destination directory"""
    for sample in sample_list:
        # Copy image file
        src_image_path = os.path.join(src_images_dir, sample)
        dest_image_path = os.path.join(dest_images_dir, sample)
        shutil.copyfile(src_image_path, dest_image_path)

        # Copy mask file (assuming the mask file has the same name as the image file)
        sample_mask = sample.replace('jpg', 'png')

        src_mask_path = os.path.join(src_masks_dir, sample_mask)
        dest_mask_path = os.path.join(dest_masks_dir, sample_mask)
        shutil.copyfile(src_mask_path, dest_mask_path)


def from_raw_masks_to_image_masks(input_dirs: list[str], output_dirs: list[str]) -> None:
    """ Converts the masks from the raw format to the image format supported by YOLOv8-Seg """
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        # Process each directories masks
        palette: list[int] = [
                0, 0, 0, # For background -> Black
                255, 0, 0, # For persons -> Red
            ]

        for j in os.listdir(input_dir):
            image_path = input_dir / j
            mask = Image.open(image_path).convert('P')
            # Ensure that all non-zero values are set to 1
            mask_data = mask.load()
            width, height = mask.size
            for y in range(height):
                for x in range(width):
                    if mask_data[x, y] > 0:
                        mask_data[x, y] = 1
            mask.putpalette(palette)
            save_path = output_dir / j
            mask.save(save_path, 'PNG')


def process_mask(image_path: str, output_file: str) -> None:
    """Process the mask image and save contours as polygons to the output file."""
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # pylint: disable = E1101
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) # pylint: disable = E1101
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # pylint: disable = E1101

    # Convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200: # pylint: disable = E1101
            polygon = []
            for point in cnt:
                polygon.append(point[0][0] / w)
                polygon.append(point[0][1] / h)
            polygons.append(polygon)

    # Save polygons to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write(f"{p}\n")
                elif p_ == 0:
                    f.write(f"0 {p} ")
                else:
                    f.write(f"{p} ")


def from_image_masks_to_labels(input_dirs: list[str], output_dirs: list[str]) -> None:
    """Converts the masks from the image format to the labels format supported by YOLOv8-Seg."""
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        for filename in os.listdir(input_dir):
            image_path = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename[:-4] + '.txt')
            process_mask(image_path, output_file)
