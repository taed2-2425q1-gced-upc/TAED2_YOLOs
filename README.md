# person_image_segmentation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         person_image_segmentation and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── person_image_segmentation   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes person_image_segmentation a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
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

The results and metrics for the model will be available in our hosted [MLFlow instance](https://dagshub.com/nachoogriis/TAED2_YOLOs.mlflow).
