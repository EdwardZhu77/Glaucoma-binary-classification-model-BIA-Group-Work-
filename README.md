# Glaucoma-binary-classification-model（BIA-Group-Work）
# Glaucoma Binary Classification Toolkit (BIA Group Work)

## Introduction
This repository contains the source code for our Biological Image Analysis (BIA) coursework. We present a comprehensive benchmarking framework for glaucoma detection using retinal fundus images. Our approach uniquely integrates visual features (via CNN backbones) with clinical metadata (Vertical Cup-to-Disc Ratio - ExpCDR) to enhance diagnostic performance.

The project implements a dual-stream approach:
1.  **Deep Learning (DL):** Using architectures like DenseNet, ResNet, and ConvNeXt.
2.  **Machine Learning (ML):** Using classifiers like XGBoost, SVM, and Random Forest.

## Project Structure
The repository is organized as follows:

glaucoma-vision/  
├── models/  
│   ├── dl/                 # Deep Learning Evaluation Scripts  
│   │   ├── evaluate_convnext.py  
│   │   ├── evaluate_densenet.py 
│   │   ├── evaluate_mobilenet.py  
│   │   └── evaluate_resnet18.py  
│   └── ml/                 # Machine Learning Evaluation Scripts  
│       ├── evaluate_rf.py         (Random Forest)  
│       ├── evaluate_svm.py        (Support Vector Machine)  
│       └── evaluate_xgb.py        (XGBoost with SHAP)  
├── utils/                  # Helper functions for data loading & preprocessing  
├── Tutorial 1...           # Guide for Image-Only Evaluation  
├── Tutorial 2...           # Guide for Hybrid (ExpCDR) Evaluation  
├── glaucoma.csv            # Clinical metadata (ExpCDR, labels)  
├── requirement list.text   # Project dependencies  
└── README.md               # Project Documentation  

## Installation and Setup

1. Clone the Repository:
   git clone https://github.com/hx03-info/Glaucoma-binary-classification-model-BIA-Group-Work-.git
   cd Glaucoma-binary-classification-model-BIA-Group-Work-

2. Install Dependencies:
   Ensure you have Python 3.8+ installed. Install the required packages:
   pip install -r "requirement list.text"

3. Dataset Preparation:
   To protect patient privacy and comply with repository size limits, the raw images are not hosted directly in this repository.
   - Download the Glaucoma Detection Dataset from Kaggle: sshikamaru/glaucoma-detection.
   - Extract the images into a local directory.
   - Ensure the file structure matches the filenames in 'glaucoma.csv'.
   - Note: You may need to update the 'BASE_PATH' variable in the evaluation scripts to point to your local image directory.

## Usage

We provide evaluation scripts for both Deep Learning and Machine Learning approaches. All models support two modes: Image-Only and Hybrid (Image + ExpCDR).

### Running Deep Learning Models
To evaluate the DenseNet-121 model (our recommended DL baseline):

python glaucoma_vision/models/dl/evaluate_densenet.py

This script will load the model, perform inference on the test set, and generate Grad-CAM visualizations to explain the model's focus areas.

### Running Machine Learning Models
To evaluate the XGBoost model with SHAP explainability:

python glaucoma_vision/models/ml/evaluate_xgb.py

This script will extract hand-crafted features (color statistics and GLCM texture descriptors), merge them with ExpCDR, and generate SHAP importance plots.

## Methodology

We implemented a benchmarking framework consisting of:

1. Deep Learning Stream:
   - Backbones: DenseNet-121, ResNet-18, ConvNeXt-Tiny, MobileNetV2.
   - Technique: Transfer learning from ImageNet with fine-tuning.
   - Hybrid Fusion: Concatenating the final dense layer embedding (1024-dim) with clinical feature embedding.
   - Explainability: Grad-CAM heatmaps.

2. Machine Learning Stream:
   - Models: XGBoost, SVM, Random Forest.
   - Features: Explicit Color histograms + GLCM Texture descriptors + ExpCDR.
   - Explainability: SHAP (TreeExplainer & KernelExplainer).

## Results

Below is a summary of our model performance on the test set:

| Model Type | Architecture | Modality | AUC Score |
| :--- | :--- | :--- | :--- |
| DL (SOTA) | DenseNet-121 | Hybrid (Img + ExpCDR) | 0.81 |
| DL | ResNet-18 | Hybrid | 0.78 |
| ML | XGBoost | Hybrid | 0.77 |
| ML | SVM | Image Only | 0.65 |

Note: Hybrid models significantly outperform Image-Only models, verifying the clinical importance of the vertical cup-to-disc ratio (ExpCDR).

## Contributors

- Zihang He: Report Discussion; Report Method
- Zhiyuan Zhu: GitHub Readme, Report Method
- Xingyang Du: Report Result; Report Method
- Haoxiang Xia: Report Result, Code Integration
- Deng Yishu: Report Introduction
- Yuang He: Report Method
