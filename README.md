# NetAurHPD - Network Auralization Hyperlink Prediction Method

## Overview
This repository contains the code for NetAurHPD model based on the paper "Network Auralization Hyperlink Prediction Model to Identify Metabolic Pathways from Metabolomics Data" by Tamir Bar-Tov, [Rami Puzis](https://scholar.google.com/citations?user=SfJ_pOYAAAAJ&hl=iw&oi=sra) and [David Toubiana](https://scholar.google.com/citations?user=-l5S-ScAAAAJ&hl=iw&oi=sra). [Link to the paper](https://arxiv.org/pdf/2410.22030)

Originaly NetAurHPD developed as a framework that relies on (1) graph auralization to extract and aggregate representations of nodes in metabolite correlation networks and (2) data augmentation method that generates metabolite correlation networks given a subset of chemical reactions defined as hyperlinks. Network Auralization is an innovative application of sound recognition neural networks to predict centrality measures of nodes within a network, by learning from the ”sound” emitted by network nodes. Networks can be likened to resonating chambers, where sound propagates through nodes and links, generating a waveform-based representation for every node. The core process of network auralization involves the propagation of energy among nodes to their neighbors until the total energy is evenly distributed throughout the entire network. In NetAurHPD we average hyperlinks waveforms to represent a hyperlink throgh a signal. Based on these hyperlinks waveforms we train M5 (very deep convolutional neural network) as classification model.

![NetAurHPD_pipeline](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/NetAurHPD_pipeline.png)

In this repository we present NetAurHPD results on common hyperlink predictions tasks as demonstrated in [A Survey on Hyperlink Prediction](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/canc/files/2207.02911.pdf)

## Components
### [Data preprocess](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/Examples/data_preprocess.py)
The `data_preprocess` and `create_train_and_test_sets` functions load and transform data into suitable training and test sets for model training.
### [Network auralization](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/network_auralization.py)
The `SignalPropagation` class implements the Network Auralization method to learn the underlying graph structure.

### [Hyperlinks waveforms](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/hyperlinks_waveforms.py)
The component averages node signals into hyperlink waveforms for further analysis.
### [M5](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/M5.py)
The M5 architecture is a very deep convolutional neural network designed for sound tasks. In this case, it is structured for binary classification tasks.

### [Predict by M5](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/predict_by_M5.py)
This module is responsible for training the M5 model and evaluating its performance on the dataset.

### [Config](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/config.py)
The `config` module contains various configurations and hyperparameters used throughout the project.

### [Utils (including Negative Sampling)](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/Examples/utils.py)
The utilities module includes the `negative_sampling` function, which generates negative hyperlinks to enhance the training dataset.
## Code Example
- [Enron dataset](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/tree/master/Examples/Enron)
- [NDC dataset](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/tree/master/Examples/NDC)

## Prerequisites
The code was implemented in python 3.9. All requirements are included in the [requirements.txt](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/requirments.txt) file.
