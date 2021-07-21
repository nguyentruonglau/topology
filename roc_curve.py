from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

import sys
import itertools
import numpy as np
import json
import argparse
import warnings
import os


def parse_args(args):
    parser = argparse.ArgumentParser("")
    parser.add_argument('--input_path', default='./output/npy_data/EfficientNetB3_bird.npy')
    parser.add_argument('--output_dir',default='./output/roc_curve')
    return parser.parse_args(args)


def roc_plot(args, y_test, y_pred, target_names, model_name):
    """Plot Receiver Operating Characteristic curve
    Args:
        args (argument parser): input information
        y_test (2D array): true label of test data
        y_pred (2D) array: prediction label of test data
        target_names (1D array): array of encode label
    Returns:
        [datagen]
    """
    n_classes = y_test.shape[1]
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    plt.figure()

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, 'roc_{}_{}.png'.format(model_name, target_names[0])))
    plt.close()

def main(args):
    #load data
    data = np.load(args.input_path, allow_pickle=True)[0]
    #get y_test, y_pred
    y_test = data['y_test']
    y_pred = data['y_pred']

    model_name, data_name = os.path.basename(args.input_path)[:-4].split('_')
    target_names=['{}'.format(data_name), 'not {}'.format(data_name)]

    roc_plot(args, y_test, y_pred, target_names, model_name)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    print('input_path=',args.input_path)
    print('output_dir=',args.output_dir)
   
    main(args)