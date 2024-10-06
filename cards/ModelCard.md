# Model Card for YOLOv8-Seg People
This model is a tuned version of YOLOv8-seg for the people segmentation task. YOLOv8-seg is a variant of the YOLO (You Only Look Once) architecture designed to perform instance segmentation. People segmentation is a task that involves identifying and delineating the regions of an image that contain people, generating accurate masks.

## Model Details
### Model Description
YOLOv8-Seg is a model built on the YOLOv8 (You Only Look Once) architecture, extending its capabilities to segment objects in an image. The model is efficient  and suitable for real-time systems that require both detection and segmentation. YOLOv8-Seg provides bounding boxes around objects while also generating pixel-level masks, allowing for precise object delineation. This model has been fine-tuned by YOLOs using the dataset_person to improve performance on specific people segmentation tasks. The additional training is focused on improving the model’s accuracy in detecting and segmenting individuals in different environments and poses.

- **Developed by:** YOLOs ( Nacho Gris, Marc Janer, Silvia Vallet, Josep Coll and Maria Risques)
- **Shared by:** Ultralytics
- **Model type:** Image Segmentation 
- **License:** Apache 2.0
- **Finetuned from model:** YOLOv8-seg (Ultralytics)

### Model Sources 
- **Repository:**
    - YOLOs: https://github.com/taed2-2425q1-gced-upc/TAED2_YOLOs
    - Ultralytics:  https://github.com/ultralytics/ultralytics
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

### Direct Use
This segmentation model, based on **YOLOv8-seg**, can be used directly for people segmentation in images and videos without the need for additional tuning. It is suitable in situations where accurate people segmentation is required without the need for additional customization or adjustments. Direct uses include:
- **Security and surveillance applications**: Real-time monitoring to detect and segment individuals in images or videos from security cameras.
- **Crowd analysis**: Identifying and tracking people at public events or crowded spaces.
- **Video and image editing**: Tools to edit or remove people from images or videos.

### Downstream Use
The model can be further tuned or integrated into broader applications to improve its performance on specific tasks. Examples of downstream uses include:
- **Custom Behavioral Analysis Systems**: By fine-tuning the model with additional data, it could be used in human behavior analysis projects, such as posture tracking or detecting specific activities.
- **Augmented Reality (AR) Applications**: Integration into AR platforms to segment and overlay people in virtual or interactive environments.
- **Mobile Applications and Edge Computing**: Tuning for mobile devices and embedded systems that require real-time processing, such as person recognition in wearable devices or smart security cameras.

### Out-of-Scope Use
The **YOLOv8-seg** model, tuned for person segmentation, is not suitable for:
- **Non-Person Object Segmentation**: Although YOLOv8 is capable of detecting multiple object classes, this model has been specifically tuned for person segmentation, so it is not suitable for general object segmentation tasks.
- **Medical or biological image processing**: The model is not trained or tuned for medical image analysis, such as organ or tissue segmentation.
- **Malicious use in unauthorized surveillance**: Use of the model in contexts that violate privacy, such as non-consensual surveillance or mass data collection, is inappropriate and may have ethical and legal consequences.

### Intended users
- **Computer vision system developers** implementing solutions for surveillance, crowd analysis, or augmented reality.
- **Computer vision researchers** interested in people segmentation for applications in motion analysis or human-computer interaction.
- **Software companies** wishing to implement video or image editing tools that require accurate people segmentation.

### Impact on affected individuals
- **Privacy**: Since the model may be used in surveillance and crowd analysis contexts, it is important to consider potential privacy concerns. Responsible use of the model should ensure compliance with data privacy regulations (e.g. GDPR).
- **Fairness**: The model may exhibit bias if it has not been trained with a sufficient diversity of images of people of different genders, races, or physical conditions. It is important to evaluate its performance in diverse settings and with diverse populations to ensure fair and equitable use.

## Bias, Risks, and Limitations
### Bias
- **Unrepresentative data**: The model has been trained on a dataset specific to people segmentation, which may introduce bias if the data does not include adequate diversity in terms of gender, race, ethnicity, or age. This could negatively impact accuracy in certain demographic groups.
- **Limited contexts**: The model may not generalize well to scenarios outside of the contexts included in the training dataset. This may include variations in lighting, complex environments, or uncommon angles.

### Risks
- **Privacy**: In applications such as surveillance or crowd analysis, use of the model may present privacy risks if not used with appropriate safeguards, such as informed consent or compliance with local regulations.
- **Misuse**: There is a risk that the model may be used in unethical ways, such as unauthorized mass surveillance or monitoring without consent.

### Limitations
- **Performance under adverse conditions**: The model may perform poorly in low-light conditions, when people are partially hidden, or when people are moving quickly.
- **Limited people segmentation**: This model is specifically tuned for people segmentation and may not be suitable for segmentation of other types of objects.

### Recommendations
- **Bias assessment**: It is recommended to evaluate the model on different demographic groups and contexts to ensure fair performance and minimize bias. The use of more diverse datasets for training and evaluation is recommended.
- **Ethical use**: Users of the model should ensure compliance with privacy and ethics regulations, especially in applications involving people surveillance or crowd analysis.
- **Adverse conditions**: To improve performance under challenging conditions, such as low lighting or complex environments, additional fine-tuning or the use of complementary image preprocessing techniques is recommended.

## How to Get Started with the Model
Use the code below to get started with the model:
```python
# Install the Ultralytics YOLOv8 package if you haven't already
!pip install ultralytics

# Import YOLOv8
from ultralytics import YOLO

# Load the fine-tuned YOLOv8-seg model
model = YOLO('model_weights.pt')  

# Segment an image
results = model('path_to_image.jpg')  

# Visualize the results
results.show()

# Optionally, save the results
results.save('path_to_save_segmented_image') 
```

## Training Details
### Training Data
YOLOv8-Seg was trained on the COCO dataset, which contains a variety of object classes such as people, vehicles, animals, and everyday objects. The dataset provides labeled bounding boxes and segmentation masks for training.

- **Dataset name:** COCO (Common Objects in Context) 
- **Classes:** 80 object classes, including people, animals, and vehicles. 
- **Data preprocessing:** Images were resized, normalized, and augmented with techniques such as random flipping, scaling, and color jittering.

Our image segmentation model, finetuned from YOLOv8-Seg, has been finetuned using data_person, a set of images featuring one or more people each. This specific dataset tries to reach a wide range of variations in people’s appearance.
- **Dataset name:** data_person
- **Classes:** 1 (people)
- **Data preprocessing:** A detailed explanation can be found in the dataset card. (FALTA LINK)

### Training Procedure
<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
#### Preprocessing 
The model was finetuned by our team on the data_person dataset, which contains over 5600 images featuring one or more people, with annotations specifically focused on person segmentation across diverse environments and scenerios. This dataset was preprocessed with several techniques to improve model performance on people segmentation tasks. 
- Annotations were transformed from pixel-wise masks into YOLO-compatible text files with polygon representations for each segmented person. 
- Augmentation techniques included flipping left-right(33% probability), scaling (±27.45%), translation (±7.69%), and HSV adjustments to hue, saturation, and value to enhance robustness. Mosaic augmentation (93.32% probability) and segment copy-paste (50% probability) were also employed to diversify image variations. 
- Data was randomly split into training (78%), validation (14%), and test (8%) sets to mitigate potential biases.

More details can be found in the dataset card (FALTA LINK).

#### Training Hyperparameters
- **Initial learning rate** (`lr0`): 0.00729
- **Final learning rate multiplier** (`lrf`): 0.00984
- **Momentum** (`momentum`): 0.98
- **Weight decay** (`weight_decay`): 0.00035
- **Warmup epochs** (`warmup_epochs`): 4.21385
- **Warmup momentum** (`warmup_momentum`): 0.76891
- **Warmup bias learning rate** (`warmup_bias_lr`): 0.1
- **Box loss gain** (`box`): 8.30896
- **Class loss gain** (`cls`): 0.56119

#### Speeds, Sizes, Times [optional]
<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->
{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation
<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics
#### Testing Data
<!-- This should link to a Dataset Card if possible. -->
{{ testing_data | default("[More Information Needed]", true)}}
#### Factors
<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->
{{ testing_factors | default("[More Information Needed]", true)}}
#### Metrics
<!-- These are the evaluation metrics being used, ideally with a description of why. -->
{{ testing_metrics | default("[More Information Needed]", true)}}
### Results
{{ results | default("[More Information Needed]", true)}}
#### Summary
{{ results_summary | default("", true) }}
## Model Examination [optional]
<!-- Relevant interpretability work for the model goes here -->
{{ model_examination | default("[More Information Needed]", true)}}
## Environmental Impact
<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->
Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).
- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}
## Technical Specifications [optional]
### Model Architecture and Objective
{{ model_specs | default("[More Information Needed]", true)}}
### Compute Infrastructure
{{ compute_infrastructure | default("[More Information Needed]", true)}}
#### Hardware
{{ hardware_requirements | default("[More Information Needed]", true)}}
#### Software
{{ software | default("[More Information Needed]", true)}}
## Citation [optional]
<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
**BibTeX:**
{{ citation_bibtex | default("[More Information Needed]", true)}}
**APA:**
{{ citation_apa | default("[More Information Needed]", true)}}
## Glossary [optional]
<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->
{{ glossary | default("[More Information Needed]", true)}}
## More Information [optional]
{{ more_information | default("[More Information Needed]", true)}}

---

## Model Card Author 
YOLOs: 
- Josep Coll
- Nacho Gris
- Marc Janer
- Maria Risques
- Silvia Vallet



## Model Card Contact
YOLOs: 
- josep.coll.sanchez@estudiantat.upc.edu
- ignacio.gris@estudiantat.upc.edu
- marc.janer@estudiantat.upc.edu
- maria.risques@estudiantat.upc.edu
- silvia.vallet@estudiantat.upc.edu
