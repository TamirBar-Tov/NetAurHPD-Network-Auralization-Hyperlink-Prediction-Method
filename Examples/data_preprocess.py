import torch

from Examples.utils import shuffle_dict, split_dictionary, merge_and_shuffle_dicts, create_y
def load_data(nodes_data_dir, groups_size_data_dir):
    """
    Loads data from specified directories for nodes and hyperlinks sizes.

    Args:
        nodes_data_dir (str): The file path to the directory containing nodes data.
        groups_size_data_dir (str): The file path to the directory containing hyperlinks size data.

    Returns:
        nodes list and sizes list:
    """
    # Read hyperlinks size data
    with open(groups_size_data_dir, "r") as file:
        hyperlinks_size_list = file.read().splitlines()

    # Read nodes list
    with open(nodes_data_dir, "r") as file:
        hyperlinks_nodes_list = file.read().splitlines()

    return hyperlinks_nodes_list, hyperlinks_size_list

def transform_data_into_hypergraph(hyperlinks_nodes_list, hyperlinks_size_list):
    """
    Transforms lists of hyperlink nodes and sizes into a unique hypergraph structure.

    Args:
        hyperlinks_nodes_list (list): List of nodes, where each node represents a hyperlink element.
        hyperlinks_size_list (list): List of integers, where each integer specifies the size of a hyperlink group.

    Returns:
        dict: A dictionary representing the unique hypergraph structure. Each key is a unique index, and each value is a sorted list of nodes in that hyperlink group.
    """
    # extract hyperlinks nodes groups
    hyperlink_dict = {}
    
    # Add hyperlinks to the graph based on the data
    current_index = 0
    for hyperlinks_group_size in hyperlinks_size_list:
        hyperlinks_group_size = int(hyperlinks_group_size)
        simplex_nodes = [int(node) for node in
                         hyperlinks_nodes_list[current_index:current_index + hyperlinks_group_size]]
        current_index += hyperlinks_group_size
        
        # delete hyperlinks with cardinality 1
        if hyperlinks_group_size < 2:
            continue
        else:
            # save in hyperlink_dict
            hyperlink_dict[len(hyperlink_dict)] = simplex_nodes

    # Total number of hyperlinks
    print("Number of hyperlinks:", len(hyperlink_dict))

    # remove duplicate hyperlinks (nodes are the key and the value)
    unique_dict = {}
    for key, value in hyperlink_dict.items():
        value_tuple = tuple(sorted(list(set(value))))
        if len(value_tuple) == 0:
            continue
        else:
            unique_dict[value_tuple] = sorted(list(set(value)))

    # save to hyperlinks to a new dictionary with numeric index
    unique_hyperlink_dict = {}
    for key, value in unique_dict.items():
        unique_hyperlink_dict[len(unique_hyperlink_dict)] = value

    return unique_hyperlink_dict

def data_preprocess(nodes_data_dir, groups_size_data_dir):
    """
    Preprocesses the data by loading and transforming it into a unique hypergraph structure.

    Args:
        nodes_data_dir (str): The file path to the directory containing nodes data.
        groups_size_data_dir (str): The file path to the directory containing hyperlink sizes data.

    Returns:
        dict: A dictionary representing the unique hypergraph structure where each key is a unique index, and each value is a list of nodes in that hyperlink group
        nodes: list of all nodes.
    """
    hyperlinks_nodes_list, hyperlinks_size_list = load_data(nodes_data_dir, groups_size_data_dir)
    unique_hyperlink_dict = transform_data_into_hypergraph(hyperlinks_nodes_list, hyperlinks_size_list)
    
    # create list of all nodes
    nodes = []
    for k,h in unique_hyperlink_dict.items():
        for node in h:
            nodes.append(node)

    nodes = list(set(nodes))
    
    return unique_hyperlink_dict, nodes

def create_train_and_test_sets(positive_hyperlink_dict, negative_hyperlink_dict):
    """
    Creates training and testing datasets by shuffling, splitting, and merging positive and negative hyperlinks.

    Args:
        positive_hyperlink_dict (dict): Dictionary of positive hyperlinks, with each key as a unique identifier and value containing details.
        negative_hyperlink_dict (dict): Dictionary of negative hyperlinks, structured similarly to `positive_hyperlink_dict`.

    Returns:
        tuple: A tuple containing:
            - train_hyperlink_dict (dict): Merged and shuffled dictionary of training hyperlinks.
            - y_train_tensor (torch.Tensor): A tensor of labels for the training set.
            - test_hyperlink_dict (dict): Merged and shuffled dictionary of testing hyperlinks.
            - y_test_tensor (torch.Tensor): A tensor of labels for the testing set.
    """
    # shuffle dictionaries
    shuffled_positive_hyperlink_dict = shuffle_dict(positive_hyperlink_dict)
    shuffled_negative_hyperlink_dict = shuffle_dict(negative_hyperlink_dict)

    # split dictionaries into train and test
    train_positive_hyperlink_dict, test_positive_hyperlink_dict = split_dictionary(shuffled_positive_hyperlink_dict)
    train_negative_hyperlink_dict, test_negative_hyperlink_dict = split_dictionary(shuffled_negative_hyperlink_dict)

    print("Train positive hyperlinks:", len(train_positive_hyperlink_dict))
    print("Train negative hyperlinks:", len(train_negative_hyperlink_dict))
    print("Test positive hyperlinks:", len(test_positive_hyperlink_dict))
    print("Test negative hyperlinks:", len(test_negative_hyperlink_dict))
    
    # merage positive and negative examples
    train_hyperlink_dict = merge_and_shuffle_dicts(train_positive_hyperlink_dict, train_negative_hyperlink_dict)
    test_hyperlink_dict = merge_and_shuffle_dicts(test_positive_hyperlink_dict, test_negative_hyperlink_dict)
    
    # create target (y)
    y_train = create_y(train_hyperlink_dict)
    y_test = create_y(test_hyperlink_dict)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)
    return train_positive_hyperlink_dict, train_hyperlink_dict, y_train_tensor, test_hyperlink_dict, y_test_tensor


