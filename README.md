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

### Setting up the environment

Before running the project, make sure that your `.env` has the same structuer as the `.env.test` file. Here is an explanation on how to come up with the necessary values for the environment variables:

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

---
