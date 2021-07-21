import numpy as np
import pandas as pd
from glob import glob
import os
from shutil import copyfile

def _read_csv(data_dir):
    directory = os.path.dirname(data_dir)
    return pd.read_csv(os.path.join(directory, 'labels.csv'))


def _filter_data(data_dir):
    print("==>begin read csv file<==")
    df_datas = _read_csv(data_dir)
    nv_image_paths = list(df_datas[df_datas['NV']==1]['path'])
    mel_image_paths = list(df_datas[df_datas['MEL']==1]['path'])

    #copy image nv to nv folder
    print("==>filter files from dataset to nv folder<==")
    for source_path in nv_image_paths:
    	dst_path = os.path.join('./data/nv', os.path.basename(source_path))
    	copyfile(source_path, dst_path)

    #copy image nv to nv folder
    print("==>filter files from dataset to mel folder<==")
    for source_path in mel_image_paths:
    	dst_path = os.path.join('./data/mel', os.path.basename(source_path))
    	copyfile(source_path, dst_path)

    return 0


if __name__ == '__main__':
	_filter_data(data_dir='data_source')