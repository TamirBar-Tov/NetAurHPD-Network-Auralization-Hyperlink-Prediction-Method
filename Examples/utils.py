import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt

from NetAurHPD.config import parse
config = parse()


def plot_results_two_lines(first_list, first_name, second_list, second_name, x_axis_name, y_axis_name, header,plot_name):
    """
    Plots two lines on a graph with specified labels and titles.
    """
    first_x_axis = list(range(len(first_list)))
    second_x_axis = list(range(len(second_list)))
    plt.plot(first_x_axis, first_list, label=first_name, color= "green")
    plt.plot(second_x_axis, second_list, label= second_name, color='gold')
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.legend()
    plt.title(header)
    plt.show()
    plt.clf()

def negative_sampling(nodes, hyperlink_dict, alpha = config.alpha):
    """
    Generates negative samples for hyperlinks, by replacing a subset of nodes with randomly selected nodes from the graph. For further explanation aboute the negative sampling method please refer to "A Survey on Hyperlink Prediction"

    Args:
        nodes (list): list of all nodes.
        hyperlink_dict (dict): Dictionary of original hyperlinks, where each key is a unique identifier, and each value is a list of nodes in the hyperlink.
        alpha (float, optional): Proportion of genuine nodes to retain in the negative samples.
    

    Returns:
        dict: A dictionary containing negative samples. Each key corresponds to a hyperlink ID, and each value is a dictionary with:
            - 'label' (str): A label indicating the sample type ('negative').
            - 'nodes' (list): A list of nodes, partially replaced with randomly chosen nodes from the graph.
    """
    negative_hyperlink_dict = {}
    for k, h in hyperlink_dict.items():
        genuineness_nodes_num = round(len(h) * alpha)
        genuineness_nodes = random.sample(h, genuineness_nodes_num)
        remaining_nodes_in_v = [x for x in nodes if x not in h]  
        additional_nodes = random.sample(remaining_nodes_in_v, len(h) - genuineness_nodes_num)
        f = additional_nodes+genuineness_nodes
        negative_hyperlink_dict[k] = {'label': 'negative', 'nodes':f}
    return negative_hyperlink_dict

def save_hyperlinks_with_label(unique_hyperlink_dict):
    """
    Adds labels to hyperlinks in a dictionary, marking each as a positive example.

    Args:
        unique_hyperlink_dict (dict): Dictionary of unique hyperlinks, where each key is a unique identifier, and each value is a list of nodes in the hyperlink.

    Returns:
        dict: A dictionary where each key corresponds to a hyperlink ID, and each value is a dictionary with:
            - 'label' (str): A label indicating the sample type ('positive').
            - 'nodes' (list): The list of nodes in the hyperlink.
    """
    positive_hyperlink_dict = {}
    for k, h in unique_hyperlink_dict.items():
        positive_hyperlink_dict[k] = {'label': 'positive', 'nodes': h}
    return positive_hyperlink_dict

def shuffle_dict(hyperlinks_dict):
    """
    Shuffles a dictionary and returns a new dictionary with random order.

    Args:
        hyperlinks_dict (dict): The dictionary to be shuffled.

    Returns:
        dict: A new dictionary with the same key-value pairs as `hyperlinks_dict`, but in a randomized order.
    """
    # Convert dictionary to list of key-value pairs
    items = list(hyperlinks_dict.items())
    # Shuffle the list
    random.shuffle(items)
    # Create a new dictionary from the shuffled list
    shuffled_dict = dict(items)
    return shuffled_dict

def split_dictionary(dict_to_split, train_size=config.train_size):
    """
    Splits a dictionary into two parts based on a specified training size ratio.

    Args:
        dict_to_split (dict): The dictionary to split.
        train_size = training size ratio
        
    Returns:
        tuple: A tuple containing two dictionaries:
            - train_part (dict): The first part of the dictionary, used for training.
            - test_part (dict): The remaining part of the dictionary, used for testing.
    """
    # Calculate the sizes of the two parts
    total_length = len(dict_to_split)
    train_length = int(config.train_size * total_length)
    # Split the dictionary into train & test parts
    train_part = dict(list(dict_to_split.items())[:train_length])
    test_part = dict(list(dict_to_split.items())[train_length:])
    return train_part, test_part

def merge_and_shuffle_dicts(pos_dict, neg_dict):
    """
    Merges two dictionaries (positive and negative samples) into one and shuffles the resulting dictionary.

    Args:
        pos_dict (dict): Dictionary containing positive samples, where each key is a unique identifier and each value is a dictionary with details.
        neg_dict (dict): Dictionary containing negative samples, structured similarly to `pos_dict`.

    Returns:
        dict: A new dictionary that contains all entries from both `pos_dict` and `neg_dict`, shuffled to randomize the order of samples.
    """
    merged_dict = {}
    for k,h in pos_dict.items():
        merged_dict[len(merged_dict)] = h
    for k,h in neg_dict.items():
        merged_dict[len(merged_dict)] = h
    merged_dict = shuffle_dict(merged_dict)
    return merged_dict

def create_y(hyperlink_dict):
    """
    Creates a list of labels (y) based on the labels in the given hyperlink dictionary.

    Args:
        hyperlink_dict (dict): Dictionary of hyperlinks, where each key is a unique identifier, and each value is a dictionary that contains a 'label' key.

    Returns:
        list: A list of integers corresponding to the labels in `hyperlink_dict`, where:
            - 0 represents a negative sample.
            - 1 represents a positive sample.
    """
    y = []
    for k,h in hyperlink_dict.items():
        if h['label'] == 'negative':
            y.append(0)
        elif h['label'] == 'positive':
            y.append(1)
        else:
            print(k)
    return y