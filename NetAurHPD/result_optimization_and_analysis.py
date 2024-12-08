from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy as np

def optimal_threshold_Youdens_J_Statistic(fpr, tpr,thresholds ):
    """
    Calculates the optimal threshold using Youden's J Statistic.
    
    Youden's J statistic is the difference between the True Positive Rate (TPR) and False Positive Rate (FPR),
    and the optimal threshold is selected as the one that maximizes this difference.
    
    Parameters:
    fpr: False Positive Rate at each threshold.
    tpr: True Positive Rate at each threshold.
    thresholds: Array of thresholds corresponding to the FPR and TPR values.
    
    Returns:
    float: The optimal threshold that maximizes the Youden's J statistic.
    """
    j_scores = tpr - fpr
    optimal_idx = j_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold



def optimize_results_and_calc_metrics(y_test,y_pred):
    """
    Optimizes the threshold using Youden's J Statistic and calculates several evaluation metrics.
    
    This function calculates the optimal threshold, AUC, accuracy, recall, precision, F1 score,
    and false positive rate (FPR) based on the true labels and predicted probabilities.
    
    Parameters:
    y_test: True binary labels for the test set.
    y_pred: Predicted probabilities or scores for the test set.
    
    Prints:
    optimal threshold, AUC, accuracy, recall, precision, F1 score, and FPR.
    """
    # calc optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, np.squeeze(y_pred))
    optimal_threshold = optimal_threshold_Youdens_J_Statistic(fpr, tpr, thresholds)
    # AUC
    roc_auc = auc(fpr, tpr)
    binary_predictions = np.squeeze((y_pred >= optimal_threshold).astype(int))
    # calc metrics
    tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
    accuracy = accuracy_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    false_positive_rate = fp / (fp + tn)
    precision = precision_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    
    print("Model Results:")
    print("optimal threshold: ",round(optimal_threshold,3))
    print("AUC: ",round(roc_auc,3))
    print("Accuracy: ",round(accuracy,3))
    print("Recall: ",round(recall,3))
    print("Precision: ",round(precision,3))
    print("F1: ",round(f1,3))
    print("FPR: ",round(false_positive_rate,3))






