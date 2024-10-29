# NetAurHPD - Network Auralization Hyperlink Prediction Method

## Overview
This repository contains the code for NetAurHPD model based on the paper "Network Auralization Hyperlink Prediction Model to
Identify Metabolic Pathways from Metabolomics Data". arxiv

Originaly NetAurHPD developed as a framework that relies on (1) graph auralization to extract and aggregate representations of nodes in metabolite correlation networks and (2) data augmentation method that generates metabolite correlation networks given a subset of chemical reactions defined as hyperlinks.

In this repository we present NetAurHPD results on common hyperlink predictions tasks as demonstrated in **survey** 

Network Auralization is an innovative application of sound recognition neural networks to predict centrality measures of nodes within a network, by learning from the ”sound” emitted by network nodes. Networks can be likened to resonating chambers, where sound propagates through nodes and links, generating a waveform-based representation for every node. The core process of network auralization involves the propagation of energy among nodes to their neighbors until the total energy is evenly distributed
throughout the entire network.

In NetAurHPD we average hyperlinks waveforms to represent a hyperlink throgh a signal. Based on these hyperlinks waveforms we train M5 (very deep convolutional neural network) as classification model.


## Components
### data preprocess
The `data_preprocess` and `create_train_and_test_sets` functions load and transform data into suitable training and test sets for model training.
### network_auralization
The `SignalPropagation` class implements the Network Auralization method to learn the underlying graph structure.

### hyperlinks_waveforms
The component averages node signals into hyperlink waveforms for further analysis.
### M5
The M5 architecture is a very deep convolutional neural network designed for sound tasks. In this case, it is structured for binary classification tasks.

### predict_by_M5
This module is responsible for training the M5 model and evaluating its performance on the dataset.

### config
The `config` module contains various configurations and hyperparameters used throughout the project.

### [utils (including Negative Sampling)](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/Examples/utils.py)
The utilities module includes the `negative_sampling` function, which generates negative hyperlinks to enhance the training dataset.
## Code Example
- [Enron dataset](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/tree/master/Examples/Enron)

