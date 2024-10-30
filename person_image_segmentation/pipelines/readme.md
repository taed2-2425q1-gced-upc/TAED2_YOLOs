# Raw Data Download Module

## Overview

This script downloads the raw data for the **Person Image Segmentation Pipeline**. It is set up to load environment variables and uses the **Kaggle API** to download the required dataset. Additionally, a test mode is available to download a smaller dataset for testing purposes.

## Script Breakdown

### 1. Imports

- **argparse**: Used for handling command-line arguments.
- **os** and **Path** from **pathlib**: For handling file paths and setting environment variables.
- **dotenv**: Used to load environment variables from a .env file.
- **person_image_segmentation.config**: Provides configuration variables such as `RAW_DATA_DIR`, `DATASET_LINK`, `KAGGLE_KEY`, and `KAGGLE_USERNAME`.
- **download_dataset**: Utility function that handles the actual dataset downloading process.

### 2. Environment Setup

The script uses the dotenv package to load environment variables from a .env file, which is necessary for accessing Kaggle credentials securely. However, this is not directly done in the `download_raw_data.py` sript, but in the `config.py` file of our module.

The variables `KAGGLE_USERNAME` and `KAGGLE_KEY` are fetched from the environment and stored into the respective environment variables to authenticate with the Kaggle API.

### 3. Command-line Arguments

The script uses **argparse** to set up command-line arguments. The main argument is:

- `--test`: If this flag is provided, the pipeline will run in test mode. In test mode, the data will be downloaded into a test directory instead of the default raw data directory.

```python
parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
```

### 4. Test Mode Logic

If the `--test` flag is present, the `RAW_DATA_DIR` is modified to point to a test directory by replacing `data` with `test_data`.

```python
if args.test:
    RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))
```

### 5. Dataset Download

Finally, the script uses the `download_dataset` function to fetch the dataset. The link to the dataset and the directory where the data will be stored are passed as parameters.

```python
download_dataset(
    dataset_link=DATASET_LINK,
    data_dir=RAW_DATA_DIR
)
```

## How to Run the Script

1. **Standard Mode**: Download the complete dataset into the raw data directory.

   ```bash
   python download_data.py
   ```

2. **Test Mode**: Download a smaller dataset for testing purposes.

   ```bash
   python download_data.py --test
   ```

## Dependencies

- **dotenv**: Used to load environment variables from .env.
- **argparse**: For handling command-line arguments.
- **Kaggle API**: Required for downloading datasets from Kaggle.

## Environment Variables

Ensure you have the following variables set in your .env file:

- `KAGGLE_USERNAME`: Your Kaggle username.
- `KAGGLE_KEY`: Your Kaggle API key.
- `RAW_DATA_DIR`: The directory where the raw data will be stored.
- `DATASET_LINK`: Kaggle dataset link for downloading.

# Data Splitting Module

## Overview

This script handles splitting the raw dataset for the **Person Image Segmentation Pipeline** into training, validation, and testing sets. The script provides the option to switch between normal and test mode to accommodate testing with a smaller dataset.

## Script Breakdown

### 1. Imports

- **argparse**: Used for handling command-line arguments.
- **person_image_segmentation.config**: Provides configuration variables like `RAW_DATA_DIR`, `SPLIT_DATA_DIR`, `TRAIN_SIZE`, `VAL_SIZE`, and `TEST_SIZE`.
- **split_dataset**: Utility function that handles the dataset splitting process into train, validation, and test sets.

### 2. Command-line Arguments

The script uses **argparse** to set up command-line arguments. The main argument is:

- `--test`: If this flag is provided, the pipeline will run in test mode. In this mode, the raw and split data directories are modified to point to test data locations.

```python
parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
```

### 3. Test Mode Logic

If the `--test` flag is present, the script modifies the `RAW_DATA_DIR` and `SPLIT_DATA_DIR` to point to the test data directories by replacing `data` with `test_data`.

```python
if args.test:
    RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))
    SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
```

### 4. Dataset Splitting

The script calls the `split_dataset` function to split the raw data into training, validation, and test sets. The split ratios are defined by the variables `TRAIN_SIZE`, `VAL_SIZE`, and `TEST_SIZE`. The raw data directory and the split data directory are passed as parameters.

```python
split_dataset(
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
    data_dir=RAW_DATA_DIR,
    split_dir=SPLIT_DATA_DIR
)
```

## How to Run the Script

1. **Standard Mode**: Split the complete dataset into train, validation, and test sets.

   ```bash
   python split_data.py
   ```

2. **Test Mode**: Split a smaller dataset for testing purposes.

   ```bash
   python split_data.py --test
   ```

## Dependencies

- **argparse**: For handling command-line arguments.
- **split_dataset**: Utility function that handles dataset splitting.

## Configuration Variables

The script uses the following configuration variables (provided in `person_image_segmentation.config`):

- `RAW_DATA_DIR`: The directory where the raw data is stored.
- `SPLIT_DATA_DIR`: The directory where the split data (train, validation, test) will be stored.
- `TRAIN_SIZE`: The ratio or percentage of data allocated for training.
- `VAL_SIZE`: The ratio or percentage of data allocated for validation.
- `TEST_SIZE`: The ratio or percentage of data allocated for testing.

Make sure these variables are properly defined before running the script.

# Mask Transformation Module

## Overview

This script transforms the masks in the dataset for the **Person Image Segmentation Pipeline**. It provides the ability to run in both normal and test modes, transforming either the full dataset or a smaller test dataset.

## Script Breakdown

### 1. Imports

- **argparse**: Used for handling command-line arguments.
- **dotenv**: Used to load environment variables from a .env file.
- **person_image_segmentation.config**: Provides configuration variables such as `SPLIT_DATA_DIR` and `TRANSFORM_DATA_DIR`.
- **transform_masks**: Utility function that handles the transformation of masks.

### 2. Command-line Arguments

The script uses **argparse** to set up command-line arguments. The main argument is:

- `--test`: If this flag is provided, the pipeline will run in test mode. In test mode, the split and transform data directories are modified to use test data directories.

```python
parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
```

### 3. Test Mode Logic

If the `--test` flag is present, the script modifies the `SPLIT_DATA_DIR` and `TRANSFORM_DATA_DIR` to point to the test data directories by replacing `data` with `test_data`.

```python
if args.test:
    SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
    TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))
```

### 4. Mask Transformation

The script calls the `transform_masks` function, which takes care of transforming the masks based on the data stored in the split data directory. The transformed masks will be saved in the transform directory.

```python
transform_masks(
    split_dir=SPLIT_DATA_DIR,
    transform_dir=TRANSFORM_DATA_DIR
)
```

## How to Run the Script

1. **Standard Mode**: Transform masks in the full dataset.

   ```bash
   python transform_masks.py
   ```

2. **Test Mode**: Transform masks in the smaller test dataset.

   ```bash
   python transform_masks.py --test
   ```

## Dependencies

- **argparse**: For handling command-line arguments.
- **transform_masks**: Utility function that handles mask transformation.

## Configuration Variables

The script relies on the following configuration variables (provided in `person_image_segmentation.config`):

- `SPLIT_DATA_DIR`: The directory where the split data is stored.
- `TRANSFORM_DATA_DIR`: The directory where the transformed masks will be saved.

Make sure these variables are properly defined before running the script.

# Label Generation Module

## Overview

This script generates labels for the **Person Image Segmentation Pipeline** based on the transformed dataset. It can be executed in either standard mode or test mode, where a smaller subset of data is used.

## Script Breakdown

### 1. Imports

- **argparse**: Used for handling command-line arguments.
- **dotenv**: Used to load environment variables from a .env file.
- \*\*

person_image_segmentation.config\*\*: Provides configuration variables such as `TRANSFORM_DATA_DIR`, `LABELS_DATA_DIR`, and `SPLIT_DATA_DIR`.

- **generate_labels**: Utility function that handles the generation of labels from the transformed data.

### 2. Command-line Arguments

The script uses **argparse** to set up command-line arguments. The main argument is:

- `--test`: If this flag is provided, the pipeline will run in test mode. In test mode, the directories for split, transformed data, and labels are modified to point to test data locations.

```python
parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
```

### 3. Test Mode Logic

If the `--test` flag is present, the script modifies the `SPLIT_DATA_DIR`, `TRANSFORM_DATA_DIR`, and `LABELS_DATA_DIR` to point to the test data directories by replacing `data` with `test_data`.

```python
if args.test:
    SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
    TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))
    LABELS_DATA_DIR = Path(str(LABELS_DATA_DIR).replace('data', 'test_data'))
```

### 4. Label Generation

The script calls the `generate_labels` function, which creates the labels based on the transformed dataset. It uses the split data, transformed data, and saves the generated labels to the designated directory.

```python
generate_labels(
    transform_dir=TRANSFORM_DATA_DIR,
    labels_dir=LABELS_DATA_DIR,
    split_dir=SPLIT_DATA_DIR
)
```

## How to Run the Script

1. **Standard Mode**: Generate labels for the full dataset.

   ```bash
   python generate_labels.py
   ```

2. **Test Mode**: Generate labels for the smaller test dataset.

   ```bash
   python generate_labels.py --test
   ```

## Dependencies

- **argparse**: For handling command-line arguments.
- **generate_labels**: Utility function that handles label generation.

## Configuration Variables

The script relies on the following configuration variables (provided in `person_image_segmentation.config`):

- `SPLIT_DATA_DIR`: The directory where the split data is stored.
- `TRANSFORM_DATA_DIR`: The directory where the transformed data is stored.
- `LABELS_DATA_DIR`: The directory where the generated labels will be saved.

Ensure these variables are properly configured before running the script.

---

# Configuration File Copy Module

## Overview

This script is responsible for copying configuration files for the **Person Image Segmentation Pipeline** from the repository to the data directory. It supports both standard and test modes, allowing users to specify a test directory for configurations.

## Script Breakdown

### 1. Imports

- **argparse**: Used for handling command-line arguments.
- **shutil**: Used for copying files.
- **Path** from **pathlib**: Used to handle file system paths.
- **person_image_segmentation.config**: Imports configuration variables `REPO_PATH` and `DATA_DIR`.

### 2. Main Execution

The script begins execution by setting up the argument parser and defining the command-line options.

```python
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()
```

### 3. Test Mode Logic

If the `--test` flag is provided, the script modifies the `DATA_DIR` to point to the test data directory by replacing `data` with `test_data`.

```python
if args.test:
    DATA_DIR = Path(str(DATA_DIR).replace('data', 'test_data'))
```

### 4. Configuration Files Copying

The script specifies a list of configuration files (`config_hyps.yaml`, `config_yolos.yaml`) to be copied from the repository's `models/configs` folder to the `DATA_DIR`. The `shutil.copy` function is used to perform the copying operation for each configuration file.

```python
config_names = ["config_hyps.yaml", "config_yolos.yaml"]

src_folder = REPO_PATH / "models/configs"
dst_folder = DATA_DIR

for config_name in config_names:
    shutil.copy(src_folder / config_name, dst_folder / config_name)
```

## How to Run the Script

1. **Standard Mode**: Copy the configuration files to the data directory.

   ```bash
   python copy_configs.py
   ```

2. **Test Mode**: Copy the configuration files to the test data directory.

   ```bash
   python copy_configs.py --test
   ```

## Dependencies

- **argparse**: For handling command-line arguments.
- **shutil**: For copying configuration files.
- **Path**: For handling file paths.

## Configuration Variables

The script relies on the following configuration variables (provided in `person_image_segmentation.config`):

- `REPO_PATH`: The path to the repository.
- `DATA_DIR`: The directory where the configuration files will be copied.

Ensure these variables are properly set up before running the script.



