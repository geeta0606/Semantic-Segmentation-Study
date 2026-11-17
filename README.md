# Comparative Study of Deep Learning Approaches for Semantic Segmentation

## Overview

This project investigates and compares different deep learning architectures for pixel-level semantic segmentation of road-scene images. The objective is to evaluate the performance of convolutional neural network-based segmentation models, instance segmentation approaches, and hybrid architectures in identifying multiple classes within complex visual scenes.

The study focuses on four semantic classes — **Car, Road, Grass, and Pole** — along with a Background class. Particular attention is given to class imbalance and the segmentation of smaller, relatively underrepresented objects such as poles.

The project explores architectural comparisons, loss function optimization, data augmentation strategies, and hybrid segmentation approaches to understand their influence on segmentation performance.

## Objectives

- Develop and compare multiple deep learning architectures for semantic segmentation.
- Investigate the impact of different loss functions on segmentation performance.
- Address class imbalance through class weighting and targeted data augmentation.
- Evaluate the effect of synthetic Pole augmentation using a cut-and-paste strategy.
- Explore prompt-based segmentation using Segment Anything Model (SAM).
- Investigate hybrid approaches combining YOLOv8 and Attention U-Net predictions.
- Perform quantitative and qualitative evaluation across multiple validation folds.

## Dataset and Preprocessing

The project uses road-scene images with corresponding colour masks and pixel-level ID masks.

Four target classes are extracted from the original label IDs:

| Original ID | Target class | Remapped ID |
|---|---|---|
| 13 | Car | 1 |
| 0 | Road | 2 |
| 9 | Grass | 3 |
| 5 | Pole | 4 |

Background is assigned ID 0, while ignored pixels retain the label 255.

### Preprocessing Pipeline

- Image and mask pairing using normalized filenames.
- Inspection of image-mask correspondence and label distributions.
- Remapping of the original segmentation labels into five target categories.
- Image resizing to 256 × 448 pixels for the Keras segmentation models.
- Normalization of image pixel values.
- Nearest-neighbour interpolation for resizing segmentation masks.
- Class-weight estimation to account for pixel-level class imbalance.

A reproducible subset of 320 images is selected using a fixed random seed and evaluated using five-fold cross-validation.

## Models and Architectures

The following approaches are investigated:

### 1. Fully Convolutional Network (FCN)
An encoder-decoder convolutional architecture that performs pixel-wise classification using convolutional and transposed convolutional layers.

### 2. U-Net
An encoder-decoder architecture incorporating skip connections to preserve spatial information and improve localization of segmented objects.

### 3. Attention U-Net
An extension of U-Net that incorporates attention gates to selectively emphasize relevant spatial features and suppress less informative regions.

### 4. YOLOv8-Seg
An instance segmentation architecture used to identify and segment individual objects through predicted bounding regions and segmentation masks. Its predictions are also converted into semantic segmentation maps for comparison.

### 5. Segment Anything Model (SAM)
SAM is investigated as a prompt-based segmentation approach using bounding boxes derived from either ground-truth annotations or YOLO predictions.

### 6. Hybrid Segmentation Approaches
Two experimental approaches are explored:

- **SAM + YOLO:** SAM generates segmentation masks using bounding boxes predicted by YOLOv8.
- **Attention U-Net + YOLO:** Pixel-wise probability maps from Attention U-Net and YOLOv8 are combined through weighted probability fusion.

## Experimental Studies

### 1. Architectural Comparison
FCN, U-Net, Attention U-Net, and YOLOv8-Seg are evaluated using the same five-fold data partitioning strategy.

### 2. Loss Function Comparison
Different loss functions are investigated using U-Net:

- Weighted Cross-Entropy
- Dice Loss
- Weighted Cross-Entropy + Dice Loss
- Focal Loss
- Focal Loss + Dice Loss
- Focal Tversky Loss

### 3. Class Imbalance and Data Augmentation
A cut-and-paste augmentation strategy is developed to increase the representation of Pole objects. Real pole regions are extracted from donor images and inserted into training images while updating their corresponding segmentation masks.

An ablation study investigates the influence of different quantities of synthetic training images.

### 4. SAM-Based Segmentation
SAM is evaluated using ground-truth bounding boxes and YOLO-predicted bounding boxes to investigate the effect of prompt quality on segmentation performance.

### 5. Probability Fusion
Attention U-Net and YOLOv8 probability maps are combined using class-wise fusion weights. The weights are tuned on validation data before evaluation on the test split.

## Evaluation Metrics

Model performance is evaluated using a confusion matrix and the following metrics:

- **Dice Coefficient:** Measures overlap between predicted and ground-truth segmentation regions.
- **Intersection over Union (IoU):** Measures the ratio of intersection to union between predicted and actual regions.
- **Precision:** Measures the proportion of correctly predicted positive pixels.
- **Recall:** Measures the proportion of actual positive pixels correctly identified.
- **Accuracy:** Measures overall pixel-wise classification accuracy.

Additional analyses include:

- True Positives, False Positives and False Negatives.
- Five-fold performance variation.
- Per-class metric comparisons.
- Qualitative prediction visualizations.
- Statistical comparisons using ANOVA and paired t-tests.

## Technologies Used

- **Programming:** Python
- **Deep Learning:** Keras, PyTorch
- **Computer Vision:** OpenCV, Pillow
- **Segmentation Models:** FCN, U-Net, Attention U-Net, YOLOv8-Seg, SAM
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Statistical Analysis:** SciPy, Statsmodels
- **Environment:** Google Colab

## Repository Structure

```text
Semantic-Segmentation-Study/
│
├── README.md
├── CV_phase2.ipynb
├── requirements.txt
├── .gitignore
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── qualitative_comparisons/
│
├── models/
│
└── docs/
    ├── methodology.md
    └── experiments.md
```

## Execution

1. Clone the repository.
2. Open the notebook in Google Colab.
3. Mount Google Drive.
4. Place the dataset in the expected directory structure.
5. Update the dataset paths in the notebook.
6. Install the required dependencies.
7. Execute the notebook cells in sequence.

**Expected dataset directory structure:**

```text
Seg_phase2/
│
├── images/
├── color_labels/
└── labels/
```

The notebook uses Google Drive for dataset access, model checkpoints and experimental result storage.

## Notes and Limitations

- The experiments use a selected subset of 320 images rather than the complete dataset.
- Five-fold cross-validation is used for model evaluation.
- Training configurations and image resolutions differ between model implementations.
- YOLOv8 and SAM require their respective pretrained model weights.
- SAM with ground-truth bounding boxes represents an experimental upper-bound-style reference and is not directly equivalent to an end-to-end inference pipeline.
- Dataset files and large model checkpoints are not included in this repository.

## Future Scope

- Evaluate the architectures on a larger dataset.
- Investigate additional augmentation techniques for underrepresented classes.
- Explore computational efficiency and inference-time comparisons.
- Improve small-object segmentation through multi-scale feature extraction.
- Investigate more robust fusion strategies for hybrid segmentation architectures.

---

**Project Focus:** Semantic Segmentation | Deep Learning | Computer Vision | Model Evaluation | Data Augmentation | Hybrid Architectures
