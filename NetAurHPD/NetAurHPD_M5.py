import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from NetAurHPD.config import parse
from Examples.utils import plot_results_two_lines

config = parse()
device = "cuda" if torch.cuda.is_available() else "cpu"

class NetAurHPD_M5(nn.Module):
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
    def __init__(self, n_input, n_output, stride=config.stride, n_channel=config.n_channel):
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
        
    def predict(self, hyperlinks_train, y_train_tensor, hyperlinks_test, y_test_tensor, lr=config.lr, total_iters = config.epochs):
        """
        Train the model using the provided training data and evaluate it on the test data.

        This method performs the following steps:
        1. Initializes the optimizer.
        2. Iterates over the specified number of epochs to train the model.
        3. Calculates and records the training and test losses.
        4. Computes and records the training and test accuracies.
        5. Plots the training and test accuracy and loss over epochs.

        Args:
            hyperlinks_train (torch.Tensor): A tensor containing the training hyperlinks.
            y_train_tensor (torch.Tensor): A tensor containing the true labels for the training data.
            hyperlinks_test (torch.Tensor): A tensor containing the test hyperlinks.
            y_test_tensor (torch.Tensor): A tensor containing the true labels for the test data.
            lr (float, optional): The learning rate for the optimizer.
            total_iters (int, optional): The total number of training iterations (epochs).

        Returns:
            numpy.ndarray: The predicted labels for the test set.
        """
        
        # settings for run
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr)#0.0001)


        losses = []
        losses_test = []
        train_accuracy_list = []
        test_accuracy_list = []
        epochs = 0

        for i in tqdm(range(epochs, epochs + total_iters)):
            optimizer.zero_grad()
            y_pred = self(hyperlinks_train.unsqueeze(dim=1).double())
            y_pred = y_pred.squeeze()
            # loss
            print(y_pred.shape)
            loss = F.binary_cross_entropy(y_pred, y_train_tensor.double())
            loss.backward()
            optimizer.step()

            losses.append(loss)

            epochs = i

            # train set accuracy
            y_pred_np = y_pred.detach().numpy()
            y_pred_classes = (y_pred_np > 0.5).astype(int)
            y_pred_classes = y_pred_classes.squeeze()
            accuracy = accuracy_score(y_train_tensor.numpy(), y_pred_classes)
            train_accuracy_list.append(accuracy)

            with torch.no_grad():
                y_pred_test = self(hyperlinks_test.unsqueeze(dim=1).double())
                y_pred_test_s = y_pred_test.squeeze()

                # test loss
                loss_test = F.binary_cross_entropy(y_pred_test_s, y_test_tensor.double())
                losses_test.append(loss_test)
                # test set accuracy
                y_pred_np_test = y_pred_test.detach().numpy()
                y_pred_classes_test = (y_pred_np_test > 0.5).astype(int)
                y_pred_classes_test = y_pred_classes_test.squeeze()
                accuracy_test = accuracy_score(y_test_tensor.numpy(), y_pred_classes_test)
                test_accuracy_list.append(accuracy_test)

        # plot accuracy train
        acc_plot_name = "accuracy"
        plot_results_two_lines(train_accuracy_list, "train_accuracy", test_accuracy_list, "test_accuracy", 'epochs', 'accuracy',
                     'accuracy',acc_plot_name)


        # loss train
        losses_list = []
        for i in range(len(losses)):
            losses_list.append(losses[i].item())

        # loss test
        losses_test_list = []
        for i in range(len(losses_test)):
            losses_test_list.append(losses_test[i].item())
        # plot loss
        plot_results_two_lines(losses_list, "train_loss", losses_test_list, "test_loss", 'epochs',
                               'loss',
                               'loss','loss')

        # max accuracy scores
        print("Max train accuracy:", max(train_accuracy_list))
        print("Max test accuracy:", max(test_accuracy_list))
        return y_pred_np_test

