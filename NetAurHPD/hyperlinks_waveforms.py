import torch
from torch_scatter import scatter_mean

def nodes_to_hyperlink(signal, hyperlink_dict, G_nodes_mapping):
    """
    Converts signals from nodes in the graph to a tensor representation a hyperlinks.

    This function takes a signal tensor and a mapping of hyperlinks, extracting the corresponding signals 
    for each node in the hyperlinks. It computes the mean signal for each hyperlink based on the signals 
    from its constituent nodes.

    Args:
        signal (torch.Tensor): A tensor containing the signals for each node in the graph.
        hyperlink_dict (dict): A dictionary mapping hyperlink IDs to their corresponding nodes.
        G_nodes_mapping (dict): A mapping from node identifiers to their indices in the signal tensor.

    Returns:
        torch.Tensor: A tensor where each row represents the mean signal vector for a hyperlink.
    """
    reactions_signals = []
    for key in hyperlink_dict.keys():
        metabolites = hyperlink_dict[key]['nodes']

        #accumulated_tensor = torch.tensor([])
        tensor_of_sequences = []
        batch = []

        for index in metabolites:
            tensor_of_sequences.append(signal[G_nodes_mapping[index]])
            batch.append(0)

        accumulated_tensor = torch.stack(tensor_of_sequences, dim=0)
        mean_vector = scatter_mean(accumulated_tensor, torch.tensor(batch), dim=0).flatten()
        reactions_signals.append(mean_vector)
        reactions_signals_tensor = torch.stack(reactions_signals)
    return reactions_signals_tensor

