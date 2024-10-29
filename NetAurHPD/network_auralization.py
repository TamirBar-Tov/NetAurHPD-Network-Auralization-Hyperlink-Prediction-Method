import torch
import networkx as nx
device = "cuda" if torch.cuda.is_available() else "cpu" 
from NetAurHPD.config import parse
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
    def __init__(self, sig, momentum=0.999, response_len=10000, tqdm=lambda x:x, device="cpu"):
        super(SignalPropagation, self).__init__()
        self._device = device
        self._a = momentum
        self._len = response_len
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

def network_auralization_from_graph(G, l=config.l):
    """
    Performs network auralization networkx graph using the SignalPropagation model.

    Args:
        G (networkx.Graph): A NetworkX graph from which to derive the adjacency matrix.
        l (int, optional): Length of the signal. 
    Returns:
        torch.Tensor: The impulse response signal for each node generated from the graph.
    """
    sig = torch.zeros(l).to(device)
    sig[0] = 1
    instrument = SignalPropagation(sig, momentum=0.999, response_len=10000, tqdm=lambda x: x, device=device).to(device)

    Adj = nx.adjacency_matrix(G).toarray()
    Adj = torch.tensor(Adj).to(device)

    # impulse response
    signal = instrument(Adj)
    return signal

