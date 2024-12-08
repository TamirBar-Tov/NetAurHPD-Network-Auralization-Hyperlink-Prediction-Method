# NetAurHPD - Network Auralization Hyperlink Prediction Method

## Overview
This repository contains the code for NetAurHPD model based on the paper "Network Auralization Hyperlink Prediction Model to Identify Metabolic Pathways from Metabolomics Data" by Tamir Bar-Tov, [Rami Puzis](https://scholar.google.com/citations?user=SfJ_pOYAAAAJ&hl=iw&oi=sra) and [David Toubiana](https://scholar.google.com/citations?user=-l5S-ScAAAAJ&hl=iw&oi=sra). [Link to the paper](https://arxiv.org/pdf/2410.22030)

Originaly NetAurHPD developed as a framework that relies on (1) graph auralization to extract and aggregate representations of nodes in metabolite correlation networks and (2) data augmentation method that generates metabolite correlation networks given a subset of chemical reactions defined as hyperlinks. Network Auralization is an innovative application of sound recognition neural networks to predict centrality measures of nodes within a network, by learning from the ”sound” emitted by network nodes. Networks can be likened to resonating chambers, where sound propagates through nodes and links, generating a waveform-based representation for every node. The core process of network auralization involves the propagation of energy among nodes to their neighbors until the total energy is evenly distributed throughout the entire network. In NetAurHPD we average hyperlinks waveforms to represent a hyperlink throgh a signal. Based on these hyperlinks waveforms we train M5 (very deep convolutional neural network) as classification model. To optimize the classification the threshold is determined using Youden's J Statistic.

![NetAurHPD_pipeline](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/NetAurHPD_pipeline.png)

In this repository we present NetAurHPD results on common hyperlink predictions tasks as demonstrated in [A Survey on Hyperlink Prediction](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/canc/files/2207.02911.pdf)

## Usage
### Method
Performing prediction using NetAurHPD requires three steps:
1. Run network auralization to find wave form to each node.
2. Average the waveforms to represent hyperlinks
3. Train M5 as classifier.
4. Optimize classification threshold and calculate score metrics.

### Configurations
Deafult configurations in [confog](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/config.py) file:
1. **alpha** = 0.5 , Proportion of genuine nodes to retain in the negative samples.
2. **beta** = 1 , ration between positiva and negative samples.
3. **l** = 10,000 , waveform length.
4. **train_size** = 0.6
5. **stride** = 8 , sliding window step in M5.
6. **n_channel** = 32 , Number of output channels for the M5 layer. 
7. **epochs** = 40 , training iterations.
8. **lr** = 0.01 , learning rate.

### Data Structure
Each sample should be in the shape of: `ID: {'label': 'positive', 'nodes': [1, 2]}`

For example:
```python
{1185: {'label': 'positive', 'nodes': [61, 108]}, 
793: {'label': 'positive', 'nodes': [58, 78]},
1139: {'label': 'negative', 'nodes': [101, 112]}}
```

### Prediction
Example showing how to perform prediction using NetAurHPD:
```python
""" required dictionaries: 
        train_positive_hyperlink_dict
        train_hyperlink_dict
        test_hyperlink_dict
        y_train (tensor)
        y_test (tensor)
        nodes list
"""

instrument = SignalPropagation(momentum=0.999, response_len=10000, tqdm=lambda x: x, device) 

train_hyperlinks_waveforms, test_hyperlinks_waveforms = instrument.networkx_auralization(
    train_positive_hyperlink_dict,train_hyperlink_dict,test_hyperlink_dict,nodes,how_graph=True)

NetAurHPD_DL_architecture = NetAurHPD_M5(n_input=1, n_output=1, stride=config.stride,
                                         n_channel=config.n_channel)
                                         
y_pred = NetAurHPD_DL_architecture.predict(train_hyperlinks_waveforms, y_train, 
                                        test_hyperlinks_waveforms, y_test,lr=config.lr,
                                        total_iters = config.epochs)
                                     
optimize_results_and_calc_metrics(y_test, y_pred)

```
### Code Examples
- [Enron dataset](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/tree/master/Examples/Enron)
- [NDC dataset](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/tree/master/Examples/NDC)

## Components
### [Data preprocess](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/Examples/data_preprocess.py)
The `data_preprocess` and `create_train_and_test_sets` functions load and transform data into suitable training and test sets for model training.

### [Network auralization](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/network_auralization.py)
The `SignalPropagation` class implements the Network Auralization method to learn the underlying graph structure. This module is also responsible for applinig auralization over networkx graph.

### [Hyperlinks waveforms](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/hyperlinks_waveforms.py)
The component averages node signals into hyperlink waveforms for further analysis.

### [NetAurHPD_M5](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/NetAurHPD_M5.py)
The M5 architecture is a very deep convolutional neural network designed for sound tasks. In this case, it is structured for binary classification tasks. This module is also responsible for training the M5 model and evaluating its performance on the dataset.

### [Result optimization and analysis](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/result_optimization_and_analysis.py)
Optimizes the threshold using Youden's J Statistic and calculates several evaluation metrics: AUC, accuracy, recall, precision, F1 score And fpr.

### [Config](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/config.py)
The `config` module contains various configurations and hyperparameters used throughout the project.

### [utils (including Negative Sampling)](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/Examples/utils.py)
The utilities module includes the `negative_sampling` function, which generates negative hyperlinks to enhance the training dataset.



## Prerequisites
The code was implemented in python 3.9. All requirements are included in the [requirements.txt](https://github.com/TamirBar-Tov/NetAurHPD-Network-Auralization-Hyperlink-Prediction-Method/blob/master/NetAurHPD/requirments.txt) file.
