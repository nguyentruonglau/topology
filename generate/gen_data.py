import numpy as np
import os
from imutils.paths import list_files
from imutils.paths import list_images
import cv2
import json
import random
import sys
import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_path', type=str, default='test')
    parser.add_argument('--output_dir_path', type=str, default='output/test')
    return parser.parse_args(args)


def list_path(dir_path):
    """List all path to directorys & image files
    """
    dir_paths = os.listdir(dir_path)
    dir_paths = np.sort(dir_paths)

    img_paths = list(list_images(dir_path))
    return dir_paths, img_paths


def choice_random(dir_path, img_paths, n):
    """Choice random image paths
    """
    random.shuffle(img_paths)
    m = 0; paths = []
    while m != n:
        rd_num = random.randint(0, len(img_paths)-1)
        if img_paths[rd_num].split('\\')[-2] != dir_path:
            paths.append(img_paths[rd_num])
            m += 1
    return paths


def main(args):
    dir_paths, img_paths = list_path(args.input_dir_path)

    map_label = dict();

    # print(img_paths)
    for idx, dir_path in enumerate(dir_paths):

        inpath = os.path.join(args.input_dir_path, dir_path)

        imgp =list(list_images(inpath))
        n = len(imgp)

        map_label[idx] = dir_path

        fname = '{}_{}.npy'.format(idx, n)
        
        data = []
        #data one
        for ip in imgp:
            img = cv2.imread(ip)
            img = cv2.resize(img, (32,32))
            data.append(img)
        #data zero
        imgp = choice_random(dir_path, img_paths, n)
        for ip in imgp:
            img = cv2.imread(ip)
            img = cv2.resize(img, (32,32))
            data.append(img)

        #save data
        file_path = os.path.join(args.output_dir_path, fname)
        np.save(file_path, data)

    with open('label_map_stanford.json', 'w') as file:
        json.dump(map_label, file)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    print('input_dir_path=', args.input_dir_path)
    print('output_dir_path=', args.output_dir_path)

    print('[INFOR]: Start...')
    main(args)
    print('[INFOR]: End')