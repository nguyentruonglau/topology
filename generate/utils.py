import matplotlib.pyplot as plt
import numpy as np
import os
import warnings


def plot_sample(data, class_name, num_img):
    """Plot sample image

    Args:
        data (4D array): image data
        class_name (string): name of class
        num_img (int): number of images used
    Returns:
        [None]
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4,4))
    ax[0].imshow(data[0])
    ax[0].set_title('{}'.format(class_name))
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(data[num_img])
    ax[1].set_title('NOT {}, EX'.format(class_name))
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.show()


def ensure_dir(directory):
    """Make sure the directory exists

    Args:
        directory (string): name of directory
         
    Returns:
        None
    """
    if not os.path.exists(directory):
        warnings.warn('''[Warning]: Output directory not found.
            The default output directory will be created.''')
        os.makedirs(directory)


def get_data(dataset_name, type_data, fname):
    """Get Cifar100 Dataset

    Args:
        fname (str): ex: 0_500 [index - number of images]
    Returns:
        [x, y]: data, label
    """
    num_img = int(fname.split('_')[-1])
    path = './generate/dataset/{}/{}/{}.npy'.format(dataset_name, type_data, fname)

    if os.path.exists(path):
        x = np.load(path)
    else: raise ValueError('[ERROR]: path to data not found')

    y = np.vstack((np.ones((num_img, 1)), np.zeros((num_img, 1))))

    print('[INFOR]: shape of x: {}'.format(x.shape))
    print('[INFOR]: shape of y: {}'.format(y.shape))
    
    #plot sample
    # plot_sample(x, fname, num_img)

    return x, y