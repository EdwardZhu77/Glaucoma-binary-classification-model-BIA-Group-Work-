# Glaucoma Binary Classification Toolkit (BIA Group Work)

## Introduction
This repository contains the source code for our Biological Image Analysis (BIA) coursework. We present a comprehensive benchmarking framework for glaucoma detection using retinal fundus images. Our approach uniquely integrates visual features (via CNN backbones) with clinical metadata (Vertical Cup-to-Disc Ratio - ExpCDR) to enhance diagnostic performance.

The project implements a dual-stream approach:
1.  **Deep Learning (DL):** Using architectures like DenseNet, ResNet, MobileNet, and ConvNeXt.
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
├── Tutorial 1: Evaluate Image-Only Models          # Guide for Image-Only Evaluation  
├── Tutorial 2：Evaluate ExpCDR with Models           # Guide for Hybrid (ExpCDR) Evaluation  
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

We evaluated multiple Deep Learning and Machine Learning architectures on the test dataset. The table below presents a comprehensive comparison of Image-Only models versus Hybrid models (denoted with the suffix "-ExpCDR", indicating the inclusion of the Vertical Cup-to-Disc Ratio).

| Model | Glaucoma_Negative F1 | Glaucoma_Positive F1 | Accuracy | AUROC Score | AUPRC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 0.76 | 0.52 | 0.68 | 0.6661 | 0.4044 |
| DenseNet121-ExpCDR | 0.58 | 0.47 | 0.53 | 0.6636 | 0.4345 |
| XGBoost | 0.77 | 0.26 | 0.64 | 0.5533 | 0.2852 |
| **XGBoost-ExpCDR** | **0.80** | **0.53** | **0.72** | **0.7672** | **0.6007** |
| SVM | 0.81 | 0.32 | 0.70 | 0.6952 | 0.4162 |
| SVM-ExpCDR | 0.81 | 0.39 | 0.72 | 0.6915 | 0.4072 |
| ResNet18 | 0.77 | 0.52 | 0.69 | 0.7056 | 0.4066 |
| ResNet18-ExpCDR | 0.79 | 0.50 | 0.71 | 0.6792 | 0.3714 |
| ConvNeXt | 0.80 | 0.51 | 0.72 | 0.7096 | 0.4381 |
| ConvNeXt-ExpCDR | 0.81 | 0.39 | 0.71 | 0.7086 | 0.3915 |
| Random Forest (RF) | 0.83 | 0.31 | 0.72 | 0.5533 | 0.3630 |
| RF-ExpCDR | 0.80 | 0.33 | 0.69 | 0.6468 | 0.3881 |
| MobileNetV2 | 0.80 | 0.43 | 0.70 | 0.7175 | 0.3926 |
| MobileNetV2-ExpCDR | 0.80 | 0.52 | 0.72 | 0.6752 | 0.3668 |

**Key Findings:**
1. **Best Overall Performance:** The XGBoost-ExpCDR model achieved the highest performance across most critical metrics, including an AUROC of 0.7672 and an AUPRC of 0.6007.
2. **Impact of Clinical Data:** The inclusion of ExpCDR significantly boosted the performance of the XGBoost model (AUROC increased from 0.55 to 0.76). However, for deep learning models like DenseNet121 and MobileNetV2, the simple concatenation of clinical data did not always yield performance improvements, suggesting the need for more complex fusion strategies or fine-tuning in future work.

## Contributors

- Zihang He: Report Discussion; Report Method
- Zhiyuan Zhu: GitHub Readme, Report Method
- Xingyang Du: Report Result; Report Method
- Haoxiang Xia: Report Result, Code Integration
- Deng Yishu: Report Introduction
- Yuang He: Report Method
