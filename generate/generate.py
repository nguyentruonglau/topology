from data import generate_binary_data_cifar10
from data import generate_binary_data_fashion_mnist
from utils import ensure_dir
import numpy as np
import tensorflow as tf
import argparse
import os


def main(FLAGS):
    num_img = FLAGS.num_img
    output_dir = FLAGS.output_dir

    #mkdir data folder, contain output
    if not os.path.exists('./data'): 
        os.mkdir('./data')

    #ensure output directory
    ensure_dir(os.path.join('./data', output_dir))

    #cifar10 dataset
    # generate_binary_data_cifar10(output_dir, num_img)

    #fashion mnist
    generate_binary_data_fashion_mnist(output_dir, num_img)

    print()


if __name__ == '__main__':

    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_img',
        type=int,
        default=1000,
        help='Number of images used'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_fashion_mnist',
        help='Path to output directory'
    )
    FLAGS = parser.parse_args()
    print("num_img = ", FLAGS.num_img)
    print("output_dir = ", FLAGS.output_dir)

    main(FLAGS)