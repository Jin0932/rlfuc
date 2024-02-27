##!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_curve)

from fashionmnist_utils import *

def balanced_accuracy(y_true, y_pred, filename):
    from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score,f1_score, roc_auc_score
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    F1 = f1_score(y_true, y_pred, average="macro") 
      
    # Convert to one-hot encoded format
    n_classes = max(max(y_true), max(y_pred)) + 1
    aa_one_hot = np.eye(n_classes)[y_true]
    bb_one_hot = np.eye(n_classes)[y_pred]

    # Calculate ROC-AUC score
    # roc_auc = roc_auc_score(aa_one_hot, bb_one_hot, multi_class='ovr')
                                                                                             
    with open(filename, 'w') as f:
        f.write("balanced_accuracy, recall, precision, F1\n")
        data = f"{balanced_accuracy:.5f} , {recall:.5f} , {precision:.5f} , {F1:.5f}\n"
        f.write(data)

    return balanced_accuracy, recall, precision, F1

def plot_normalized_confusion_mtx(y_true, y_predicted, filename):
    normalized_confusion_mtx = confusion_matrix(y_true, y_predicted, normalize='true')
    sns.set(style="white")
    cmap = sns.color_palette("Blues")
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(normalized_confusion_mtx, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5)
    ax.set_title(dataset_name+' Dataset Normalized Confusion Matrix\n', fontsize=16)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(filename, dpi=600)
    # plt.show()


def write_true_prediction(y_true, y_pred, filename):
    with open(filename, 'w') as f:
        f.write("True, prediction\n")
        data = f"{y_true} , {y_pred}\n"
        f.write(data)
