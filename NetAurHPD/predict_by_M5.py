import torch
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from NetAurHPD.M5 import M5
from NetAurHPD.config import parse
config = parse()
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def predict(hyperlinks_train, y_train_tensor, hyperlinks_test, y_test_tensor, lr=config.lr, total_iters = config.epochs):
    """
    Trains a binary classification model on the training dataset and evaluates its performance on the test dataset.
    This function performs the following steps:
    1. Initializes NetAurHPD model and the optimizer.
    2. Trains the model for a specified number of epochs while recording the training and test losses.
    3. Calculates and stores the training and test accuracies after each epoch.
    4. Plots the training and test accuracies and losses.
    5. Prints the maximum training and test accuracies achieved during training.
    
    Args:
        hyperlinks_train (torch.Tensor): Training data tensor of hyperlinks.
        y_train_tensor (torch.Tensor): Tensor containing the true labels for the training data.
        hyperlinks_test (torch.Tensor): Test data tensor of hyperlinks.
        y_test_tensor (torch.Tensor): Tensor containing the true labels for the test data.
        lr (float): learning rate.
        total_iters (int): epochs.
    """
    # settings for run
    NetAurHPD = M5(n_output=1).to(device)
    optimizer = torch.optim.Adam(NetAurHPD.parameters(), lr)#0.0001)


    losses = []
    losses_test = []
    train_accuracy_list = []
    test_accuracy_list = []
    epochs = 0

    for i in tqdm(range(epochs, epochs + total_iters)):
        optimizer.zero_grad()
        y_pred = NetAurHPD(hyperlinks_train.unsqueeze(dim=1).double())
        y_pred = y_pred.squeeze()
        # loss
        loss = binary_cross_entropy(y_pred, y_train_tensor.double())
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
            y_pred_test = NetAurHPD(hyperlinks_test.unsqueeze(dim=1).double())
            y_pred_test_s = y_pred_test.squeeze()

            # test loss
            loss_test = binary_cross_entropy(y_pred_test_s, y_test_tensor.double())
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





