# Person Image Segmentation with YOLOv8-Seg finetuned version

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The main objective of the project is to build and deploy a machine learning (ML) component, using and following software engineering best practices.

More specifically, an image segmentation problem with focus on human figure detection using a You Only Look Once  model has been selected, a fine-tuned YOLO v8-seg to be precise.

The primary aim is to develop a robust system that not only detects but also accurately segments human figures in various image sets. By refining the YOLO v8-seg model, we seek to achieve high precision in recognizing and delineating human shapes amidst other objects in the frame.

Hence, the end goal will be for the model to successfully detect and segment people within an input set of images containing various elements.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
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
├── tests
├── metrics         
├── cards            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── ModelCard.md    
│   └── DatasetCard.md            
│
└── person_image_segmentation   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes person_image_segmentation a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train_v0.py            <- Code to train model v0
    │   └── evaluation.py            <- Code to evaluate models
    ├── pipelines 
    │   └── dataPipelines.py 
    ├── utils 
```

--------

