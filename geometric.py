from numpy.linalg import norm
import numpy as np 
from random import gauss
import copy
from scipy.spatial.distance import cdist


def gram_schmidt(X):
    """Calculate gram schmidt

    Args:
        X [2D array]: matrix of points, each point is a column
    Returns:
        [Q]: gram schmidt of X
    """

    Q, R = np.linalg.qr(X)
    return Q


def random_unit_vector(dims):
    """Randomly generates a vector unit with dims dimensions

    Args:
        dims [int]: dims dimensions
    Returns:
        []: unit vector
    """

    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def normalizate_min_max(data, new_min=0, new_max=1):
    """Min-Max normalization

    Args:
        data (2D array): data to normal
        new_min (int): new min
        new_max (int): new max

    Returns:
        [2D array]: data after min-max normalization
    """
    _min = data.min()
    _max = data.max()
    data = ((data-_min)/(_max - _min)) * (new_max - new_min) + new_min
    return data


def normalizate_density(density):
    """Normalize the probability density values

    Args:
        density (1D array): probability density values
    
    Returns:
        [1D array]: the probability density values have been standardized
    """
    density = np.array(density)
    probability = density / np.sum(density)
    return probability


def calculate_total_norm(F, P1, P2, S1, S2, R=None, FLAG='before_transform'):
    """Calculate total variance norm before and after applying matrix R

    Args:
        F (2D array): feature vectors
        P1, P2: probability density function
        S1 (2D array): 1D -> [feature vector, true label]
        S2 (2 D array): 1D -> [feature vector, true label one]
        R (2D array): matrix geometric transform
    
    Returns:
        [float]: total variance norm
    """

    #get D1
    density_one = np.exp(P1.score_samples(S1)) #*
    density_one = density_one.reshape((density_one.shape[0], -1))
    density_one = normalizate_density(density_one)

    #get D2
    density_two = np.exp(P2.score_samples(S2)) #*
    density_two = density_two.reshape((density_two.shape[0], -1))
    density_two = normalizate_density(density_two)

    #define
    total_norm = 0; proba_two=None

    if FLAG == 'before_transform':
        #get total norm
        for d1, d2 in zip(density_one, density_two):
            sub = d1[0] - d2[0]
            if sub >= 0: total_norm += sub
        return total_norm

    elif FLAG == 'after_transform':
        #D1xR
        density_one = np.hstack((F, density_one))
        density_one = density_one.dot(R)
        proba_one = density_one[:,-1]

        #D2xR
        density_two = np.hstack((F, density_two))
        density_two = density_two.dot(R)
        proba_two = density_two[:,-1]

        for pro_one, pro_two in zip(proba_one, proba_two):
            sub = pro_one - pro_two
            if sub >= 0: total_norm += sub

        return total_norm, proba_two
    else: raise ValueError('[ERROR]: FLAG not found')


def get_matrix_geometric_transformation(F, S1, S2, P1, P2):
    """Get matrix R of geometric transformation

    Args:
        F (2D array): feature vectors
        S1 (2D array): feature vectors + predict label
        S2 (2D array): feature vectors + true label
        P1 (pdf object): probability density function of S1, [filtered one]
        P2 (pdf object): probability density function of S2

    Returns:
        [2D array]: matrix R, maxtric transform
    """
    print('[INFOR]: Start create matrix geometry transform R')
    num_dims = F.shape[1] #dimension of feature vector
    #random n_k vector unit
    unit_vectors = []
    for i in range(2000):
        temp = random_unit_vector(dims=(num_dims+1)); unit_vectors.append(temp)
    unit_vectors = np.array(unit_vectors)

    #create u_i
    density_one = np.exp(P1.score_samples(S1)); density_one = normalizate_density(density_one)
    density_two = np.exp(P2.score_samples(S2)); density_two = normalizate_density(density_two)

    #different
    diff_pdf = density_one - density_two
    diff_pdf = diff_pdf.reshape((diff_pdf.shape[0], -1))
    
    #add diff_pdf to S1
    us = np.hstack((F, diff_pdf))

    #calculation
    ls_mul_positives = []
    for unit_vector in unit_vectors:
        sum_positive = 0
        for ui in us:
            temp = unit_vector.dot(ui)
            if temp>0: sum_positive += temp
        ls_mul_positives.append(sum_positive)

    #selected
    index = np.argmin(ls_mul_positives)
    nk_selected = unit_vectors[index]

    #gram-schmidt
    matrix_onehot = np.eye((num_dims+1), dtype = float)

    us = np.vstack((nk_selected, matrix_onehot))
    us = us.T
    output = gram_schmidt(us)

    #remove min values after gram_schmidt
    output = output.T
    ls_norms = [norm(out) for out in output]
    index = np.argmin(ls_norms)
    output = list(output[0:index]) + list(output[index+1:])
    output = np.asarray(output)

    #result
    output = np.vstack((output, nk_selected))
    R = output.T
    print('[INFOR]: Shape of matrix R: {}x{}'.format(R.shape[0], R.shape[1]))
    return R