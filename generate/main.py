import numpy as np
import os
from utils import get_data
import sys
import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar_100', help='Choice: [cifar_100, fgvc_aircraft, stanford_car]')
    parser.add_argument('--type_data', type=str, default='train', help='Choice: [train, test]')
    parser.add_argument('--show', type=bool, default=True)
    parser.add_argument('--fname', type=str, default='0_500')
    return parser.parse_args(args)

def main(args):
    x, y = get_data(args.dataset_name, args.type_data, args.fname)
   
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    print('dataset_name=', args.dataset_name)
    print('type_data=', args.type_data)
    print('show=', args.show)
    print('fname=', args.fname)

    print('[INFOR]: Start...')
    main(args)
    print('[INFOR]: End')