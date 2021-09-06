from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from imutils.paths import list_files
from sklearn.metrics import auc

import sys
import itertools
import numpy as np
import json
import argparse
import warnings
import os


def parse_args(args):
    parser = argparse.ArgumentParser("")
    parser.add_argument('--input_dir', default='./output/npy_data')
    parser.add_argument('--output_auc_dir',default='./output/auc_data')
    return parser.parse_args(args)


def roc_plot(args, alpha, precision_before, sensitivity_before, precision_after, sensitivity_after, roc_name):
    """Plot Receiver Operating Characteristic curve
    Args:
        args (argument parser): input information
        alpha (1D array): ex: [0.01, ..., 0.99]
        sensitivity, specificity, precision (1D array)
        roc_name (str): nam of roc curve
    Returns:
        None
    """
    # Plot all ROC curves
    plt.figure()
    lw = 2

    colors = ['darkorange', 'cornflowerblue']
    #sensitivity
    plt.plot(precision_before, sensitivity_before, lw=lw, color= colors[0], label='Before, area = {}'.format(auc(precision_before, sensitivity_before)))
    #precision
    plt.plot(precision_after, sensitivity_after, lw=lw, color= colors[1], label='After, area = {}'.format(auc(precision_after, sensitivity_after)))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Sensitivity')
    # plt.title('Roc_{}.png'.format(roc_name))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_auc_dir, 'Auc_{}.png'.format(roc_name)))
    plt.close()


def main(args):
    ls_paths = list(list_files(args.input_dir))

    for path in ls_paths:
        roc_name = os.path.basename(path)[:-4]

        if roc_name.startswith('skin_cancer'): num_img_per_class = 450
        else: num_img_per_class = 1000

        #load data
        data = np.load(path, allow_pickle=True)[0]

        #testing
        proba_zero = np.array(data['proba_one_testing'])
        proba_one_before = np.array(data['proba_two_testing_before'])
        proba_one_after = np.array(data['proba_two_testing_after'])

        #update proba_zero following: a > alpha (a+b) -> assign = 1
        proba_zero = proba_zero + proba_one_before

        #alpha = [0.01, ..., 0.99]
        alpha = np.arange(0,1, 0.01); alpha = np.delete(alpha, 0)
        sensitivity_before = []; precision_before = []

        #before
        for alp in alpha:
            scores = [] #ex: scores = [1, 1, 0, 1, ...]
            for pbo, pbz in zip(proba_one_before, proba_zero):
                if pbo > alp*pbz: scores.append(1)
                else: scores.append(0)
            #get sensitivity, specificity & precision
            sen = round((np.sum(scores[0: num_img_per_class]) / num_img_per_class), 2)
            pre = round((np.sum(scores[0:num_img_per_class]) / (np.sum(scores) + 1e-6)), 2)

            #append
            sensitivity_before.append(sen)
            precision_before.append(pre)

        #update proba_zero following: a > alpha (a+b) -> assign = 1
        proba_zero = proba_zero + proba_one_after
        sensitivity_after = []; precision_after = []

        #after
        for alp in alpha:
            scores = [] #ex: scores = [1, 1, 0, 1, ...]
            for pbo, pbz in zip(proba_one_after, proba_zero):
                if pbo > alp*pbz: scores.append(1)
                else: scores.append(0)
            #get sensitivity, specificity & precision
            sen = round((np.sum(scores[0: num_img_per_class]) / num_img_per_class), 2)
            pre = round((np.sum(scores[0:num_img_per_class]) / (np.sum(scores) + 1e-6)), 2)

            #append
            sensitivity_after.append(sen)
            precision_after.append(pre)

        print('alpha: ', alpha)

        print('sensitivity_before: ', sensitivity_before)
        print('sensitivity_after: ', sensitivity_after)
        print('precision_before: ', precision_before)
        print('precision_after: ', precision_after)

        sensitivity_before = np.sort(sensitivity_before)
        sensitivity_after = np.sort(sensitivity_after)
        precision_before = np.sort(precision_before)
        precision_after = np.sort(precision_after)

        print('sensitivity_before: ', sensitivity_before)
        print('sensitivity_after: ', sensitivity_after)
        print('precision_before: ', precision_before)
        print('precision_after: ', precision_after)
        roc_plot(args, alpha, precision_before, sensitivity_before, precision_after, sensitivity_after, roc_name)
        break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    print('input_dir=',args.input_dir)
    print('output_auc_dir=',args.output_auc_dir)
    
    print('[INFOR]: Starting drawing...')
    main(args)
    print('[INFOR]: Completion!')