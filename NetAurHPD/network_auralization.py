import torch
import networkx as nx
from NetAurHPD.hyperlinks_waveforms import nodes_to_hyperlink
device = "cuda" if torch.cuda.is_available() else "cpu" 
from NetAurHPD.config import parse
import random
from itertools import combinations
import matplotlib.pyplot as plt

config = parse()

class SignalPropagation(torch.nn.Module):
    """
    A PyTorch module that simulates signal propagation through graph using a specified response function.

    Args:
        sig (torch.Tensor): A tensor representing the signal response over time.
        momentum (float, optional): Momentum factor to stabilize updates (default is 0.999).
        response_len (int, optional): Length of the response sequence to simulate (default is 10000).
        tqdm (callable, optional): A function for displaying progress (default is a no-op lambda).
        device (str, optional): Device to run the computation on, e.g., 'cpu' or 'cuda' (default is 'cpu').

    Attributes:
        _device (str): The device on which computations are performed.
        _a (float): Momentum factor.
        _len (int): Length of the response sequence.
        _sig (torch.Tensor): The input signal tensor.
        _sig_loop (int): The length of the signal tensor.
        _tqdm (callable): Progress display function.

    Methods:
        forward(A):
            Propagates the signal through the adjacency matrix `A` and returns the response over time.

        """
    def __init__(self, momentum=0.999, response_len=10000, tqdm=lambda x:x, device="cpu"):
        super(SignalPropagation, self).__init__()
        self._device = device
        self._a = momentum
        self._len = response_len
        sig = torch.zeros(config.l).to(device)
        sig[0] = 1        
        self._sig = sig.to(self._device)
        self._sig_loop = len(sig)
        self._tqdm = tqdm


        
    def forward(self, A):
        n  = len(A)
        P = torch.ones((n,n)).to(self._device)
        Snew = Sold = S = torch.zeros((1,n)).to(self._device)
        dS = dSold = torch.zeros((n,n)).to(self._device)
        M = torch.zeros((n,n)).to(self._device)

        response = []
        for i in self._tqdm(range(0,self._len)):
            Sold = S
            S = Snew
            dSold = dS
            S += self._sig[i % (self._sig_loop-1)]
            response.append(S)
            dS = (A / (A.sum(dim=0)+1E-32)).T
            D = torch.diag(S.squeeze())
            dS = D.mm(dS) + self._a*M
            Snew = S + dS.sum(dim=0) - dS.sum(dim=1)
            M = dS
        response = torch.stack(response).squeeze().T
        return response

    def networkx_auralization(self,train_positive_hyperlink_dict,train_hyperlink_dict,test_hyperlink_dict,nodes, show_graph=True):
        """
        Performs auralization on a networkx graph from the training hyperlinks and generates waveforms for both training and test hyperlinks.

        This method performs the following steps:
        1. Constructs a networkx graph using the clique expansion transformation based on the positive training hyperlinks.
        2. Creates a mapping of graph nodes to their indices.
        3. Computes the adjacency matrix of the graph and converts it to a PyTorch tensor.
        4. Calculates the impulse response of the graph using the model.
        5. Generates waveforms for the training and test hyperlinks based on the computed signal.

        Args:
            train_positive_hyperlink_dict (dict): A dictionary containing positive training hyperlinks.
            train_hyperlink_dict (dict): A dictionary containing all training hyperlinks.
            test_hyperlink_dict (dict): A dictionary containing test hyperlinks.
            nodes (list): A list of nodes in the graph.
            show_graph (bool, optional): If True, displays the constructed graph. Default is True.

        Returns:
            train_hyperlinks_waveforms (numpy.ndarray): The generated waveforms for the training hyperlinks.
            test_hyperlinks_waveforms (numpy.ndarray): The generated waveforms for the test hyperlinks.

        """
        G = self.clique_expansion_transformation(train_positive_hyperlink_dict,nodes, show_graph)
        G_nodes_mapping = {}
        for i in G.nodes():
            G_nodes_mapping[i] = len(G_nodes_mapping)

        Adj = nx.adjacency_matrix(G).toarray()
        Adj = torch.tensor(Adj).to(device)

        # impulse response
        signal = self(Adj)
        train_hyperlinks_waveforms = nodes_to_hyperlink(signal, train_hyperlink_dict, G_nodes_mapping)
        test_hyperlinks_waveforms = nodes_to_hyperlink(signal, test_hyperlink_dict, G_nodes_mapping)
        
        return train_hyperlinks_waveforms, test_hyperlinks_waveforms
    
    @staticmethod
    def clique_expansion_transformation(train_positive_hyperlink_dict,nodes, show=True):
        """
        Transforms a hypergraph into a graph using clique expansion, where each hyperlink is expanded into a fully connected subgraph.

        Args:
            train_positive_hyperlink_dict (dict): Dictionary representing the hypergraph structure, where each key is a positive hyperlink index in the train set, and each value is a list of nodes in that hyperlink.
            nodes (list): list of all nodes
            show (bool, optional): Whether to visualize the graph. Defaults to True.

        Returns:
            networkx.Graph: A NetworkX graph object where each hyperlink is transformed into a clique of nodes.

        Visualization:
            If `show` is True, displays a visual representation of the graph using Matplotlib.
        """

        train_positive_hyperedge_dict_for_G = {}
        for k,h in train_positive_hyperlink_dict.items():
            train_positive_hyperedge_dict_for_G[len(train_positive_hyperedge_dict_for_G)] = train_positive_hyperlink_dict[k]['nodes']
            # Create an empty graph
            G = nx.Graph()

        for key, value in train_positive_hyperedge_dict_for_G.items():
            simplex_nodes = value
            G.add_nodes_from(simplex_nodes)
            G.add_edges_from(combinations(simplex_nodes, 2))

        # we add all nodes to G, even those without connections
        G.add_nodes_from([x for x in nodes if x not in G.nodes()])

        # Print nodes and edges of the graph
        print("Nodes:", len(G.nodes()))
        print("Edges:", G.size())
        print("HyperLinks:", len(train_positive_hyperedge_dict_for_G))

        if show:
            # Visualize the graph (optional)
            plt.figure(figsize=(20, 20))
            nx.draw_kamada_kawai(G, with_labels=True)
            plt.show()
        return G

