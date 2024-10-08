import argparse

from person_image_segmentation.config import REPO_PATH, DATA_DIR


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        DATA_DIR = Path(str(DATA_DIR).replace('data', 'test_data'))
    
    config_names = ["config_hyps.yaml", "config_yolos.yaml"]
    
    src_folder = REPO_PATH / "models/configs"
    dst_folder = DATA_DIR

    for config_name in config_names:
        shutil.copy(src_folder / config_name, dst_folder / config_name)