# Person Image Segmentation with YOLOv8-Seg finetuned version

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The main objective of the project is to build and deploy a machine learning (ML) component, using and following software engineering best practices.

More specifically, an image segmentation problem with focus on human figure detection using a You Only Look Once model has been selected, a fine-tuned YOLO v8-seg to be precise.

The primary aim is to develop a robust system that not only detects but also accurately segments human figures in various image sets. By refining the YOLO v8-seg model, we seek to achieve high precision in recognizing and delineating human shapes amidst other objects in the frame.

Hence, the end goal will be for the model to successfully detect and segment people within an input set of images containing various elements.

## Project Organization

The `data` folder that appears in this section is not in the GitHub repository, but will be generated once the `dvc repro` command ends.

```
├── LICENSE            <- Apache 2.0. License
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── configs
│   └── weights
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `0_NG_initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         person_image_segmentation and configuration for tools like black
├── poetry.lock
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│   └── cards
│       ├── ModelCard.md
│       └── DatasetCard.md
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── tests
│
├── metrics
│
├── frontend           <- Code for the UI used to test our ML component
│
├── static             <- Folder to store generated documents (cleaned periodically except for the favicon.ico)
│
├── person_image_segmentation   <- Source code for use in this project.
│   │
│   ├── __init__.py             <- Makes person_image_segmentation a Python module
│   │
│   ├── config.py               <- Store useful variables and configuration
│   │
│   ├── api
│   │   ├── app.py              <- Code to define the FastAPI application and its endpoints
│   │   └── schema.py           <- Code to define data models used for responses
│   │
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── predict.py          <- Code to run model inference with trained models
│   │   ├── train_v0.py         <- Code to train model v0
│   │   ├── simple_train.py     <- Code to perform a simple training
│   │   └── evaluation.py       <- Code to evaluate models
│   │
│   ├── pipelines
│   │   ├── download_raw_data.py        <- Code to download raw data
│   │   ├── split_data.py               <- Code to split data
│   │   ├── transform_masks.py          <- Code to transform masks to YOLO format
│   │   ├── create_labels.py            <- Code to create labels
│   │   └── complete_data_folder.py     <- Code to copy files
│   │
│   └── utils
│
├ .env.test                     <- Sample .env file with the main structure
│
├ dvc.lock                      <- Stores status of DVC pipeline previous executions
│
└ dvc.yaml                      <- Yaml file describing stages, commands, dependencies and outputs of the DVC pipeline
```

## Instructions

Please, follow the instructions below to run the project:

### Cloning the repository

Clone this repository to be able to run the project locally. You can do this by running:

```bash
git clone https://github.com/taed2-2425q1-gced-upc/TAED2_YOLOs.git
cd TAED2_YOLOs
```

### Installing dependencies and libraries

Once you have cloned the project into you local machine, you need to install the necessary dependencies and libraries.

#### Installing poetry

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and libraries. Make sure you have it installed on you machine by running:

```bash
poetry --version
```

If you don't have Poetry installed, you can install it by running:

```bash
pipx install poetry
```

For more information on how to get started with Poetry, refer to the [official documentation](https://python-poetry.org/docs/).

#### Installing dependencies

Once you have Poetry installed, you can install the project's dependencies by running:

```bash
poetry install
```

This will install all the necessary dependencies and libraries for the project.

### Setting up the environment

Before running the project, make sure that your `.env` has the same structure as the `.env.test` file. Here is an explanation on how to come up with the necessary values for the environment variables:

- `KAGGLE_USERNAME` -> Your Kaggle username
  - Login to Kaggle and navigate to your profile page
- `KAGGLE_KEY` -> Your Kaggle API key
  - Navigate to Settings -> Account -> API
  - Click on "Create New API Token"
  - Copy the API key
- `MLFLOW_TRACKING_USERNAME` -> Your MLFlow username (you may use your DagsHub username)
- `MLFLOW_TRACKING_PASSWORD` -> A personal access token or API key generated in DagsHub
- `PATH_TO_DATA_FOLDER` -> The path to the `data` folder
  - In the root of the project, create a folder called `data`
  - Right click on the folder name and select "Copy path"
- `PATH_TO_REPO` -> The path to the root of the project
  - It should look something like this: `.../.../TAED2_YOLOs`

Note that there are some variables with a default value. You should not modify them, as they apply to all users running the project.

### Running the pipeline

All the stages of the project are integrated into a single dvc pipeline. Thus, you can run the entire pipeline by running:

```bash
dvc repro
```

This will run all the stages of the pipeline and create the necessary files in the `data` folder.

Keep in mind that this pipeline is configured to run the training with a single epoch, so the weights generated will not be final and should not be considered the best results of the model.

> **Note**: The training stage can be executed using a GPU. However, some specific GPUs (Apple, for example) are not detected. In that case, the training stage can take more or less 30'.

Additionally, in the evaluation stage, only 10 images will be used to speed up the process. This means that each time the pipeline is run, the evaluation results may vary, since the images selected could be different in each run.

This configuration is intended to allow the pipeline to run completely without requiring too much time or resources.

The results and metrics for the model will be available in our hosted [MLFlow instance](https://dagshub.com/nachoogriis/TAED2_YOLOs.mlflow).

> **Note**: During this process, human authorization will be needed for Dagshub. This is necessary to complete the model run. Make sure you keep an eye on notifications to grant the necessary permissions.

Additionally, to make the pipeline run faster, we have configured it to train using the test dataset instead of the full training dataset. The test dataset contains significantly fewer images, allowing us to verify that the pipeline works correctly on local machines without long wait times. Keep in mind that this setup is intended for quick testing and does not reflect the final model's performance.

### Running the training in kaggle to get the final model

This steps are further steps to train a complete model. Make sure to execute them if a complete training process is expected.

#### Upload the Dataset to Kaggle

To upload the dataset to Kaggle, run the following command:

```bash
python3 run_kaggle_dataset.py
```

This command will upload the previously configured dataset to your Kaggle account.

#### Upload the Model to Kaggle and Run

To upload the model to Kaggle and run the training, use the following command:

```bash
python3 run_kaggle_model.py
```

> **Note**: During this process, Kaggle will ask for human authorization on the platform for DagsHub. This is necessary to complete the model run. Make sure you keep an eye on Kaggle notifications to grant the necessary permissions.

This last notebook will end up as soon as the training is remotely started at kaggle. However, the training will continue. In real time tracking can be done from the DagsHub experiments section.
