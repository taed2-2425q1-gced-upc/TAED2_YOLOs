# Dataset Card for Dataset Person

## Dataset Description

- **Homepage**: [TAED2_YOLOs Dataset](https://www.kaggle.com/datasets/mariarisques/dataset-person-yolos)
- **Repository**: [TAED2_YOLOs](https://github.com/taed2-2425q1-gced-upc/TAED2_YOLOs.git)
- **Point of Contact**: Javier Ruiz Hidalgo (UPC Professor)

### Dataset Summary

This dataset contains a set of images in which zero, one or more people appear. This dataset, might also include other elements that are not people. The goal by using this dataset is to fine-tune a previously existing instance segmentation model in order to generate masks corresponding to people in any image. It will be our objective to avoid segmenting as well other objects that are not people. This dataset can be used for different tasks such as surveillance applications, video and image editing, etc., among others.

This dataset has been generating from a raw set of images (most of them containing people). This dataset tries to reach a wide range of variations in people's appearance (man, woman, whit people, black people, etc.).

### Supported tasks

This dataset supports a wide range of tasks. In this section all these tasks will be detailed.

- **Person segmentation**: The dataset can be used to train a model for person segmentation, which is the process of partitioning an image into different segments, assigning each pixel to a specific object or region (which in this case would be people). Success on this task is typically measured by achieving a high `IoU` (Intersection over Union).

- **Person detection**: The dataset can be used to train a model for person detection, which is a computer vision task focused on identifying and locating human figures within images or video frames, often providing bounding boxes around each detected person. Success on this task is typically measured by achieving a high `mAP` (often `mAP:0.5:0.95` is used, to average the results of different thresholds).

- **Pose estimation**: The dataset can be used to train a model for pose estimation, a computer vision task that involves detecting and estimating the orientation and position of a person's body in an image or video. The goal is to identify key points on the body, such as joints or limbs, to construct a representation of the person's pose. Success on this task is typically measured by achieving a high `PCK` (Percentage of Correct Keypoints) or `OKS` (Object Keypoint Similarity).

- **Action recognition**: The dataset can be used to train a model for action recognition, a computer vision task that involves identifying and classifying specific actions or activities performed by individuals in images or videos, such as walking, running or jumping. Success is typically measured by achieving a high `Accuracy` or `F1`.

### Languages

This is an instance segmentation dataset. Therefore, no languages appear in it.

## Data Structure

### Data instances
This dataset contains a total of 5639 images and, therefore, a total of 5639 labels. 

Each instance can be divided in two parts: the sample and the label. The sample will correspond to the image we want to segment. The label will correpsond to a `txt` file with `N` lines. Each line will relate to a segmented person.

There is no relation among different instances. Each instance has its own people to segment and no relations apply. An error in a segmentation will not affect another sample prediction.

The sample of a data instance can be described using JSON format in the following way.

````
{
    'image': 'path_to_the_image'
}
````

The label of a data instance can be described using JSON format in the following way.

````
{
    'label': {
        'class_1 polygon_1_point_1 polygon_1_point_2 ... polygon_1_point_M1',
        'class_2 polygon_2_point_1 polygon_2_point_2 ... polygon_2_point_M2',
        ...
        'class_N polygon_N_point_1 polygon_N_point_2 ... polygon_N_point_MN',
    }
}
````

This is a JSON format representation. The real dataset will not be in this format. Each segmentation line (`N` different of these) is formed by a class and a polygon. In our case, as we are finetuning an image segmentation model to segment a single class (person), there will only be a single class and, therefore, this number will allways be the same, a zero. The polygon is represented by `Mi` points (being `i` the number of the segmented person from `[1,N]`).

### Data Fields

As mentioned before, there are two different data fields. Each of them will be contained in a different file (one for the sample image and one for the label).

- `image`: The image that will be used for people segmentation. Its data type, once read by the computer, will be an array of integer values representing the pixels of the image (3 dimensional array as we have to take into account the RGB channels). This corresponds to the input of the model in all tasks.

- `label`: The label that will be used to evaluate the segmentation. Its data type, once read by the computer, will be a string out of which we will be able to extract the segmentation class and the masked polygon. This corresponds to the output of the model in all tasks.

### Data splits

This dataset contains different splits of data. Particularly, there are three different splits: train split, validation split and test split.

All splits have been designed in a way that possible biases and differences among splits are mitigated. This has been done by randomly sorting the images before generating the splits. This ensures that no previous order can alter the expected results by setting a certain type of person completely in a split and a different certain type of person in another one.

- Training split: This training split is used to train and update the model's parameters. The training split contains ~78% of the total data instances.

- Validation split: This validation split is used to validate the model while training. The metrics corresponding to this dataset are the ones that, in the end, are used to choose among different training experiments. This dataset can be considered as a simulation from a real-life test. The data instances included in this split are used for training, so they must never be considered as a test metric. The validation split contains ~14% of the total data instances.

- Test split. This tet split is used to test the final chosen model once this one has been chosen. It will be used to evaluate the model's performance on a set of images it has never seen. The data instances included in this split must never be used to train the model. The test split contains ~8% of the total data instances.

Below, some general statistics for each test split can be found summarized in the following table.

| Feature                  | Train Split | Validation Split | Test Split |
| :----------------------: | :---------: | :--------------: | :--------: |
| Number of Samples        | 4395        | 775              | 469        |
| Total Number of People   | TODO        | TODO             | TODO       |
| Average Number of People | TODO        | TODO             | TODO       |

## Dataset Creation

### Curation Rationale

This dataset has been created in to be able to provide a robust, complete and consistent dataset for people-related computer vision tasks. By combining previously existing datasets, it is ensured that a big diversity is presented in the dataset, thus mitigating possible biases in the data.

### Source Data

As it has been mentioned, the dataset has multiple sources. Particularly, the dataset has been created by obtaining random samples of data from the following resources.

- [Human3.6m](http://vision.imar.ro/human3.6m/description.php)
- [Multi-Human Parsing in the Wild](https://arxiv.org/pdf/1705.07206)
- [OCHuman](https://github.com/liruilong940607/OCHumanApi/tree/master)
- [COCO-Human](https://github.com/cocodataset)

#### Initial Data Collection and Normalization

The initial data collection process has consisted, as it has been already stated, on taking random samples from the previous resources and merging them together to provide a complete, robust and diverse dataset. This dataset has been, in addition, randomly sorted to avoid possible patterns in data than may affect the training process.

No normalization steps have been necessary to generate the dataset.

### Annotations

This dataset contained some initial annotations that some models such as YOLOv8-seg do not understand. The initial annotations are images with the same dimensions as the sample images whose pixels' values are 0 if that pixel in the sample image corresponds to background and a positive integer if the pixel in the sample image corresponds to people (That positive integer will indicate to which person that pixel corresponds to).

#### Annotation process

**Finish when the annotation process has been finished, so all steps are clear.**

In order to re-anotate the dataset samples to achieve some annotations that YOLO models can read we have to transform the label images to label text files in a specific format. To do so, we have used some python scrips (_**Link to the folder with the different python scripts for data transformations**_) that read the actual annotated images, extract the information and generate a text file for each of the data instances in our dataset.

This process has been required for each of the data instances of the dataset. The validation annotation process has been implemented by checking the resulting mask as an image and comparing it to the original image.

#### Who are the annotators?

The anotators are the members of the YOLOs group.

- Josep Coll
- Ignacio Gris
- Marc Janer
- Maria Risques
- Silvia Vallet

### Personal and Sensitive Information

In the dataset there are no categories that could be considered to identify any people. However, in some images where the face of the people appearing in the image is visible, some cross-referencing with other datasets containing the person's name or some other personal information might allow to identify that person.

This data does not include any type of information that could be considered sensitive, as the only category available is `person`. However, it is true that some images' background could be used to identify the location where the images were taken, despite being difficult.

## Considerations for Using the Data

### Social Impact of Dataset

This image segmentation dataset, focused on detecting the full bodies of people, could contribute to advancements in public safety, crowd management, and pedestrian detection for autonomous systems. These improvements may enhance emergency response and urban planning. However, the use of such data carries significant risks, including privacy concerns, especially when applied in surveillance contexts. There is also a risk of reinforcing biases if the dataset lacks sufficient diversity in terms of race, gender, or geography, potentially leading to unfair outcomes, such as biased policing. Additionally, if the dataset includes images from low-resource or underrepresented communities, it may expose these groups to greater risks of surveillance or unintended consequences, underscoring the need for cultural sensitivity and privacy protections.

### Discussion of Biases

This dataset, which focuses on detecting full-body images of people, may reflect several potential biases. One common bias could be demographic underrepresentation, where certain groups—such as people of specific races, genders, or age groups—are either overrepresented or underrepresented. This can lead to models that perform poorly on less-represented groups, perpetuating inequalities in applications like surveillance or pedestrian detection.

Geographic bias is another concern. If the dataset primarily includes images from certain regions, such as urban areas in developed countries, models trained on this data may not generalize well to rural or low-resource environments. Similarly, social and cultural contexts might be misrepresented or ignored if the data lacks diversity in attire, behavior, or settings.

No specific steps may have been taken to address these biases unless actively mitigated through diverse data collection or balancing techniques. If bias analysis has not been performed on this dataset, such efforts would be critical to ensure more equitable outcomes and reduce the risk of harm to marginalized communities in real-world applications.

### Other known limitations

No other known limitations arise from this dataset, as no studies have worked on it and no other information is available.

## Additional Information

### Dataset Curators

The people involved in collecting the dataset are the following:

- Josep Coll
- Ignacio Gris
- Marc Janer
- Maria Risques
- Silvia Vallet

All these people belong to YOLOs.

### Licensing Information

No license applies to this dataset.

### Citation Information

If you use this dataset, please cite as follows.

````
@misc{
    author = {Josep Coll, Ignacio Gris, Marc Janer, Maria Risques, Silvia Vallet},
    title  = {Dataset Person},
    year   = {2025}
}
````

### Contributions

Thanks to Javier Ruiz Hidalgo for providing us with the raw dataset.
