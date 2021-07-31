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
    parser.add_argument('--input_dir', default='./output/npy_data')
    parser.add_argument('--output_roc_dir',default='./output/roc_curve')
    parser.add_argument('--output_data_dir',default='./output/data_roc')
    return parser.parse_args(args)


def roc_plot(args, alpha, sensitivity, specificity, precision, roc_name):
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

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    #sensitivity
    plt.plot(alpha, sensitivity, lw=lw, color= colors[0], label='Sensitivity')
    #specificity
    plt.plot(alpha, specificity, lw=lw, color= colors[1], label='specificity')
    #precision
    plt.plot(alpha, precision, lw=lw, color= colors[2], label='precision')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Alpha')
    # plt.ylabel('True Positive Rate')
    plt.title('Roc_{}.png'.format(roc_name))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_roc_dir, 'Roc_{}.png'.format(roc_name)))
    plt.close()


def main(args):
    ls_paths = list(list_files(args.input_dir))

    for path in ls_paths:
        save_data = dict()
        roc_name = os.path.basename(path)[:-4]

        if roc_name.startswith('skin_cancer'): num_img_per_class = 450
        else: num_img_per_class = 1000

        #load data
        data = np.load(path, allow_pickle=True)[0]

        #testing
        proba_zero = np.array(data['proba_one_testing'])
        proba_one = np.array(data['proba_two_testing_after'])

        #update proba_zero following: a > alpha (a+b) -> assign = 1
        proba_zero = proba_zero + proba_one

        #alpha = [0.01, ..., 0.99]
        alpha = np.arange(0,1, 0.01); alpha = np.delete(alpha, 0)
        sensitivity = []; specificity = []; precision = []

        for alp in alpha:
            scores = [] #ex: scores = [1, 1, 0, 1, ...]
            for pbo, pbz in zip(proba_one, proba_zero):
                if pbo > alp*pbz: scores.append(1)
                else: scores.append(0)
            #get sensitivity, specificity & precision
            sen = np.sum(scores[0: num_img_per_class]) / num_img_per_class
            spe = (num_img_per_class - np.sum(scores[num_img_per_class:])) / num_img_per_class
            pre = np.sum(scores[0:num_img_per_class]) / np.sum(scores)

            #append
            sensitivity.append(sen)
            specificity.append(spe)
            precision.append(pre)

        roc_plot(args, alpha, sensitivity, specificity, precision, roc_name)

        save_data['sensitivity'] = sensitivity
        save_data['specificity'] = specificity
        save_data['precision'] = precision

        save_data = [save_data]
        np.save(os.path.join(args.output_data_dir, 'Measure_{}.npy'.format(roc_name)), save_data)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    print('input_dir=',args.input_dir)
    print('output_roc_dir=',args.output_roc_dir)
    print('output_data_dir=',args.output_data_dir)
    
    print('[INFOR]: Starting drawing...')
    main(args)
    print('[INFOR]: Completion!')