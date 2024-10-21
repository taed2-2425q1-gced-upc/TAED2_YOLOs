""" Module with processing utils """
import shutil
import os
from cv2 import imread, threshold, findContours, contourArea # pylint: disable=E0611
from cv2 import  RETR_EXTERNAL, CHAIN_APPROX_SIMPLE # pylint: disable=E0611
from cv2 import IMREAD_GRAYSCALE, THRESH_BINARY # pylint: disable=E0611
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


def from_image_masks_to_labels(input_dirs: list[str], output_dirs: list[str]) -> None:
    """ Converts the masks from the image format to the labels format supported by YOLOv8-Seg """
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        for j in os.listdir(input_dir):
            image_path = os.path.join(input_dir, j)
            # load the binary mask and get its contours
            mask = imread(image_path, IMREAD_GRAYSCALE)
            _, mask = threshold(mask, 1, 255, THRESH_BINARY)

            h, w = mask.shape
            contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

            # convert the contours to polygons
            polygons = []
            for cnt in contours:
                if contourArea(cnt) > 200:
                    polygon = []
                    for point in cnt:
                        x, y = point[0]
                        polygon.append(x / w)
                        polygon.append(y / h)
                    polygons.append(polygon)

            # print the polygons
            with open(
                f'{os.path.join(output_dir, j)[:-4]}.txt', 'w', encoding='utf-8'
            ) as f:
                for polygon in polygons:
                    for p_, p in enumerate(polygon):
                        if p_ == len(polygon) - 1:
                            f.write(f'{p}\n')
                        elif p_ == 0:
                            f.write(f'0 {p} ')
                        else:
                            f.write(f'{p} ')
            f.close()
