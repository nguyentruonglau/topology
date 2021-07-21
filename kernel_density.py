from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from geometric_transform import normalizate_min_max
import numpy as np


def get_pdf_positive(model, num_image, x, y, u):
    """Use KDE to get PDF for positive label

    Args:
        model (model object): model of last layer (get feature vector)
        num_image (int): number of image for one class in binary task
        x (4D array): [NxHxWxC]
        u (2D array): [N,1] - prediction, ex: [0,1]
        y (2D array): [N,1] - label, ex: [0,1]

    Returns:
        [F: features, S1c, S2c: data, P1, P2: pdf]
    """
    print('\n[INFOR]: Create PDF for Positive Label\n')
    #get feature vector
    F = model.predict(x, verbose=1, workers=2, use_multiprocessing=True)

    # project to a lower dimension
    pca = PCA(n_components=16, whiten=False)
    F = pca.fit_transform(F)

    #normalizate data
    F = normalizate_min_max(F, new_min=0, new_max=1)
    
    #S1 = horizontal stack feature + y
    mask = y.reshape((y.shape[0], ))
    S1 = np.hstack((F, y))
    SC1 = S1.copy()
    S1 = S1[mask==1]

    #S2 = horizontal stack feature + u
    mask = u.reshape((u.shape[0], )); mask = mask[0:num_image];
    S2 = np.hstack((F, u))
    SC2 = S2.copy()
    S2 = S2[0:num_image]
    S2 = S2[mask==1]
    
    #get kde S1
    P1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(S1)
    #get kde S2
    P2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(S2)
    return F, SC1, SC2, P1, P2


def get_pdf_negative(model, num_image, x, y, u):
    """Use KDE to get PDF for negative label

    Args:
        model (model object): model of last layer (get feature vector)
        num_image (int): number of image for one class in binary task
        x (4D array): [NxHxWxC]
        u (2D array): [N,1] - prediction, ex: [0,1]
        y (2D array): [N,1] - label, ex: [0,1]

    Returns:
        [features]: 
    """
    print('\n[INFOR]: Create PDF for Negative Label\n')
    #get feature vector
    F = model.predict(x, verbose=1, workers=2, use_multiprocessing=True)

    # project to a lower dimension
    pca = PCA(n_components=16, whiten=False)
    F = pca.fit_transform(F)

    #normalizate data
    F = normalizate_min_max(F, new_min=0, new_max=1)
    
    #S1 = horizontal stack feature + y
    mask = y.reshape((y.shape[0], ))
    S1 = np.hstack((F, y))
    SC1 = S1.copy()
    S1 = S1[mask==0]

    #S2 = horizontal stack feature + u
    mask = u.reshape((u.shape[0], )); mask = mask[num_image:];
    S2 = np.hstack((F, u))
    SC2 = S2.copy()
    S2 = S2[num_image:]
    S2 = S2[mask==0]
    
    #get kde S1
    P1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(S1)
    #get kde S2
    P2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(S2)
    return F, SC1, SC2, P1, P2