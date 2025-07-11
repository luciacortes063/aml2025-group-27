# Graph Neural Network-Based Classification of Alzheimer’s Disease Stages Using rs-fMRI Data

## 1. Objective
The goal of this project is to develop a machine learning model that classifies brain scans into three categories of cognitive state:
- Cognitively Normal (CN)
- Mild Cognitive Impairment (MCI)
- Alzheimer’s Disease (AD)

We aim to use Graph Neural Networks (GNNs), specifically Graph Attention Networks (GATs), to incorporate both spatial and functional connectivity information derived from resting-state functional MRI (rs-fMRI) data. Pretrained Convolutional Neural Networks (CNNs) will be used for feature extraction.
The motivation behind this project is to contribute, even in a small way, to the early detection of Alzheimer's disease, which is essential for extending the life expectancy of those who suffer from it or are likely to develop it.

##  2. Dataset
We will use the publicly available Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. For this project, we will focus on rs-fMRI scans.
To simplify processing, we may use standard brain atlases (e.g., AAL atlas) to define brain regions as nodes.

We will perform a stratified data split to ensure that each class (CN, MCI, AD) is proportionally represented in the training, validation, and test sets. We will use a 70-15-15 split or 5-fold stratified cross-validation depending on the final dataset size after preprocessing.

## 3. Methodology

### 3.1 Preprocessing of the rs-fMRI scans

(Slice Timing Correction, Motion Correction,  Spatial Smoothing, Denoising, etc  )

### 3.2 Feature Extraction

Each subject’s brain is segmented into predefined regions of interest (ROIs).

For each region, features will be extracted either from 2D image slices using a pretrained CNN (maybe ResNet-18), or directly from the functional connectivity profile of that region (depends on what gives better performance).


This results in a feature vector per region (node in the graph).


### 3.3 Graph Construction
Nodes represent brain regions.


Edges are defined based on anatomical adjacency or functional correlation between regions.


The resulting graph captures the structural or functional topology of each subject's brain.


### 3.4 Graph Neural Network
We apply a stack of Graph Attention Network (GAT) layers to learn how relevant regions interact.


Global pooling (e.g., mean or max) is applied to produce a graph-level embedding.


A fully connected layer (MLP) maps this embedding to one of the three classes (CN, MCI, AD).

### 3.5 Tuning the model
We will perform a grid search over the number of GAT layers using the validation set. Due to time and resource constraints, we will restrict the search to a few reasonable combinations.


## 4. Evaluation Protocol
 The model will be evaluated using:
- Accuracy

### 4.1 Baselines
- *Simple Machine Learning Baseline*: We will compare results with a machine learning baseline, using a Multilayer Perceptron (MLP).
- *Statistical Baseline*: To assess whether our GNN model achieves meaningful performance beyond trivial classifiers, we will use a Random Classifier.  
- *Literature Baseline*:  Finally, we will compare our results with the literature on Alzheimer’s disease classification using rs-fMRI and brain functional networks, in particular the article “Multi-label classification of Alzheimer's disease stages from resting-state fMRI-based correlation connectivity data and deep learning”, which serves as a performance benchmark for similar tasks using deep learning.

## Team Members:
- **Lucía Cortés** - [https://github.com/luciacortes063](https://github.com/luciacortes063)
- **Álex Capilla** - [https://github.com/AlexCapillaUZH](https://github.com/AlexCapillaUZH)
- **Sergi Garcia** - [https://github.com/sgm17](https://github.com/sgm17)
