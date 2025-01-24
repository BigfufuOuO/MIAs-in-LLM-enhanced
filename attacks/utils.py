import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

def draw_auc_curve(fpr, tpr,
                   title='ROC curve',
                   save_path='./results'):
    """
    Draw the ROC curve and save it to the save_path.
    """
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # draw title
    plt.title(title)
    
    # save the figure
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))