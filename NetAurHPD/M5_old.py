import torch
from torch import nn
import torch.nn.functional as F
from NetAurHPD.config import parse
config = parse()

class M5(nn.Module):
    """
    A 1D CNN for binary classification.

    This model consists of multiple convolutional layers followed by batch normalization and pooling layers.
    It is designed to take 1D input data and produce an output representing the probabilities of the classes.

    Args:
        n_input (int): Number of input channels.
        n_output (int): Number of output classes.
        stride (int): The stride of the sliding window. 
        n_channel (int): Number of output channels for the first convolutional layer. 

    Methods:
        forward(x): Passes the input through the network and returns the predicted probabilities.
    """
    def __init__(self, n_input=1, n_output=2, stride=config.stride, n_channel=config.n_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(2*n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(2*n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2*n_channel, n_output)
        self.double()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return torch.sigmoid(x)


