from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from imutils.paths import list_files

import sys
import itertools
import numpy as np
import json
import argparse
import warnings
import os


def parse_args(args):
    parser = argparse.ArgumentParser("")
    parser.add_argument('--input_dir', default='./output/data_roc/cifar10/resnet50')
    parser.add_argument('--output_dir',default='./output/paper_roc')
    parser.add_argument('--num_class',type=int, default=10)
    return parser.parse_args(args)


def roc_plot_sensitivity(args, sensitivity, label):
    """Plot Receiver Operating Characteristic curve
    Args:
        args (argument parser): input information
        alpha (1D array): ex: [0.01, ..., 0.99]
        sensitivity, specificity, precision (1D array)
        file_name (str): nam of roc curve
    Returns:
        None
    """
    # Plot all ROC curves
    plt.figure()
    lw = 2; num_class = args.num_class
    alpha = np.arange(0,1, 0.01); alpha = np.delete(alpha, 0)

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'deepskyblue', 'darkcyan', 'violet', 'fuchsia', 'lightpink', 'lime', 'mediumblue']
    #sensitivity
    for i in range(num_class):
        plt.plot(alpha, sensitivity[i], lw=lw, color= colors[i],
                     label=label[i]
            )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Alpha')
    # plt.ylabel('True Positive Rate')
    model_name = args.input_dir.split('/')[-1]
    data_name = args.input_dir.split('/')[-2]
    title = 'Sensitivity - {} - {}'.format(data_name, model_name)
    file_name = 'sensitivity{}_{}'.format(data_name, model_name)
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.output_dir, 'Roc_{}.png'.format(file_name)))
    plt.close()


def roc_plot_specificity(args, specificity, label):
    """Plot Receiver Operating Characteristic curve
    Args:
        args (argument parser): input information
        alpha (1D array): ex: [0.01, ..., 0.99]
        sensitivity, specificity, precision (1D array)
        file_name (str): nam of roc curve
    Returns:
        None
    """
    # Plot all ROC curves
    plt.figure()
    lw = 2; num_class = args.num_class
    alpha = np.arange(0,1, 0.01); alpha = np.delete(alpha, 0)

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'deepskyblue', 'darkcyan', 'violet', 'fuchsia', 'lightpink', 'lime', 'mediumblue']

    #specificity
    for i in range(num_class):
        plt.plot(alpha, specificity[i], lw=lw, color= colors[i],
                     label=label[i]
            )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Alpha')
    # plt.ylabel('True Positive Rate')
    model_name = args.input_dir.split('/')[-1]
    data_name = args.input_dir.split('/')[-2]
    title = 'Specificity - {} - {}'.format(data_name, model_name)
    file_name = 'specificity_{}_{}'.format(data_name, model_name)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(args.output_dir, 'Roc_{}.png'.format(file_name)))
    plt.close()


def roc_plot_precision(args, precision, label):
    """Plot Receiver Operating Characteristic curve
    Args:
        args (argument parser): input information
        alpha (1D array): ex: [0.01, ..., 0.99]
        sensitivity, specificity, precision (1D array)
        file_name (str): nam of roc curve
    Returns:
        None
    """
    # Plot all ROC curves
    plt.figure()
    lw = 2; num_class = args.num_class
    alpha = np.arange(0,1, 0.01); alpha = np.delete(alpha, 0)

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'deepskyblue', 'darkcyan', 'violet', 'fuchsia', 'lightpink', 'lime', 'mediumblue']

    #precision
    for i in range(num_class):
        plt.plot(alpha, precision[i], lw=lw, color= colors[i],
                     label=label[i]
            )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Alpha')
    # plt.ylabel('True Positive Rate')
    model_name = args.input_dir.split('/')[-1]
    data_name = args.input_dir.split('/')[-2]
    title = 'Precision - {} - {}'.format(data_name, model_name)
    file_name = 'precision{}_{}'.format(data_name, model_name)
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.output_dir, 'Roc_{}.png'.format(file_name)))
    plt.close()


def main(args):
    ls_paths = list(list_files(args.input_dir))
    sensitivity = []; specificity = []; precision = []
    labels = []

    for path in ls_paths:
        
        #load data
        data = np.load(path, allow_pickle=True)[0]

        #get sen, spe, pre
        sen = data['sensitivity']
        spe = data['specificity']
        pre = data['precision']

        #append
        sensitivity.append(sen)
        specificity.append(spe)
        precision.append(pre)

        label = os.path.basename(path).split('_')[-1].split('.')[0]
        labels.append(label)

    roc_plot_sensitivity(args, sensitivity, labels)
    roc_plot_specificity(args, specificity, labels)
    roc_plot_precision(args, precision, labels)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    print('input_dir=',args.input_dir)
    print('output_dir=',args.output_dir)
    print('num_class=',args.num_class)
    
    print('[INFOR]: Starting drawing...')
    main(args)
    print('[INFOR]: Completion!')