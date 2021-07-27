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


def get_data_cifar(data_name, num_img=1000, class_name='airplane'):
    """Get Cifar10 Dataset

    Args:
        num_img (int): number of images for a class to be taken

        [1: 'airplane', 2: 'automobile', 3: 'bird', 4: 'cat', 5: 'deer', \
        6: 'dog', 7: 'frog', 8: 'horse', 9: 'ship', 10: 'truck']

        class_name (tring): name of class

    Returns:
        [x, y]: data, label
    """
    path = './generate/dataset/{}/{}.npy'.format(data_name, class_name)
    if os.path.exists(path):
        x = np.load(path)
    else: raise ValueError('[ERROR]: path to data not found')

    y = np.vstack((np.ones((num_img, 1)), np.zeros((num_img, 1))))

    print('[INFOR]: shape of x: {}'.format(x.shape))
    print('[INFOR]: shape of y: {}'.format(y.shape))
    
    #plot sample
    # plot_sample(x, class_name, num_img)

    return x, y


def get_data_mnist(data_name, num_img=1000, class_name='t-shirt'):
    """Get Fashion Mnist Dataset

    Args:
        num_img (int): number of images for a class to be taken

        [1: 'T-shirt/top', 2: 'Trouser', 3: 'Pullover', 4: 'Dress', 5: 'Coat', \
        6: 'Sandal', 7: 'Shirt', 8: 'Sneaker', 9: 'Bag', 10: 'Ankle boot']

        class_name (string): name of class

    Returns:
        [x, y]: data, label
    """
    path = './generate/dataset/{}/{}.npy'.format(data_name, class_name)
    if os.path.exists(path):
        x = np.load(path)
    else: raise ValueError('[ERROR]: path to data not found')

    y = np.vstack((np.ones((num_img, 1)), np.zeros((num_img, 1))))

    print('[INFOR]: shape of x: {}'.format(x.shape))
    print('[INFOR]: shape of y: {}'.format(y.shape))

    #plot sample
    # plot_sample(x, class_name, num_img)
    
    return x, y


def get_data_skin_cancer(data_name, num_img, class_name='melanoma'):
    """Get Skin Cancer Dataset

    Args:
        class_name (string): name of class

        [1: 'melanoma', 2: 'nevus']

    Returns:
        [x, y]: data, label
    """

    #check path
    path_one = './generate/dataset/{}/melanoma.npy'.format(data_name)
    path_two = './generate/dataset/{}/nevus.npy'.format(data_name)
    if not os.path.exists(path_one) or not os.path.exists(path_two):
        raise ValueError('[ERROR]: path to data not found')

    num_img = None
    #load data
    if class_name == 'melanoma':
        class_one = np.load(path_one)
        class_two = np.load(path_two)
        x = np.vstack((class_one, class_two))
        y = np.vstack((np.ones((len(class_one), 1)), np.zeros((len(class_one), 1))))
    else:
        class_one = np.load(path_two)
        class_two = np.load(path_one)
        x = np.vstack((class_one, class_two))
        y = np.vstack((np.ones((len(class_one), 1)), np.zeros((len(class_one), 1))))


    print('[INFOR]: shape of x: {}'.format(x.shape))
    print('[INFOR]: shape of y: {}'.format(y.shape))

    #plot sample
    # plot_sample(x, class_name, num_img)
    
    return x, y