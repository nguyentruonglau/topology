from geometric import get_matrix_geometric_transformation
from geometric import calculate_total_norm
from generate.utils import get_data_mnist, get_data_cifar
from generate.utils import get_data_skin_cancer
from utils import kde_plot, get_model
from utils import pre_processing_data
from kernel_density import get_pdf_positive
from kernel_density import get_pdf_negative
from geometric import normalizate_density

from utils import model_from_layer
from utils import get_recall
from utils import get_predict
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from geometric import normalizate_min_max

import time
import os
import numpy as np
import tensorflow as tf
import argparse
import copy

from tensorflow.keras.utils import to_categorical

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def main(FLAGS):
    
    #define number of images
    MODEL_NAME = FLAGS.model_name
    CLASS_NAME = FLAGS.class_name
    DATA_NAME = FLAGS.data_name
    DATA_SHAPE = tuple(FLAGS.data_shape)
    MODEL_SHAPE = tuple(FLAGS.model_shape)
    SAVE_DATA = dict()

    # ==================== TRAIN ====================
    print('==================== TRAINING STAGE ====================')

    NUM_IMG_TRAIN = FLAGS.num_img_train
    
    #get data train
    x_train, y_train = get_data_cifar(data_name='cifar10', num_img=NUM_IMG_TRAIN, class_name=CLASS_NAME)
    
    #get model
    model = get_model(model_name=MODEL_NAME, data_shape=DATA_SHAPE, model_shape=MODEL_SHAPE)

    #preprocessing data
    x_train = pre_processing_data(model_name = MODEL_NAME, data = x_train)

    #get predicted_of_model
    predicted_of_model = get_predict(model, x_train, num_imgs=NUM_IMG_TRAIN)

    #get last layer
    model_k = model_from_layer(model, name_layer='avg_pool')

    #estimate pdf - positive
    F, S1O, S2O, P1O, P2O = get_pdf_positive(model_k, NUM_IMG_TRAIN, x_train, y_train, predicted_of_model)

    # ===== P(x,1) BEFORE =====
    density_two_before = np.exp(P2O.score_samples(S2O))
    proba_two_before = normalizate_density(density_two_before)

    #calculate total norm
    total_norm = calculate_total_norm(F, P1O, P2O, S1O, S2O, FLAG='before_transform')
    print('[INFOR]: Toltal norm before transform: {:.4f}'.format(total_norm))

    #get matrix R
    R = get_matrix_geometric_transformation(F, S1O, S2O, P1O, P2O)

    # ===== P(x,1) AFTER =====
    total_norm, density_two_after = calculate_total_norm(F, P1O, P2O, S1O, S2O, R=R, FLAG='after_transform')
    proba_two_after = normalizate_density(density_two_after)
    print('[INFOR]: Toltal norm after transform: {:.4f}'.format(total_norm))

    #estimate pdf - negative
    _, _, S2Z, _, P2Z = get_pdf_negative(model_k, NUM_IMG_TRAIN, x_train, y_train, predicted_of_model)

    # ===== P(x,0) =====
    density_one = np.exp(P2Z.score_samples(S2Z))
    proba_one = normalizate_density(density_one)

    #get recall before transform
    recall_class_one, recall_class_zero = get_recall(proba_one, proba_two_before, NUM_IMG_TRAIN)
    SAVE_DATA['proba_one_training'] = proba_one
    SAVE_DATA['proba_two_training_before'] = proba_two_before

    print('[INFOR]: Recall training of class {} before transform: {}'.format(CLASS_NAME, recall_class_one))
    print('[INFOR]: Recall training of class not {} before transform: {}\n'.format(CLASS_NAME, recall_class_zero))

    SAVE_DATA['recall_one_training_before'] = recall_class_one
    SAVE_DATA['recall_zero_training_before'] = recall_class_zero

    #get recall after transform
    recall_class_one, recall_class_zero = get_recall(proba_one, proba_two_after, NUM_IMG_TRAIN)

    SAVE_DATA['proba_two_training_after'] = proba_two_after

    print('[INFOR]: Recall training of class {} after transform: {}'.format(CLASS_NAME, recall_class_one))
    print('[INFOR]: Recall training of class not {} after transform: {}'.format(CLASS_NAME, recall_class_zero))

    SAVE_DATA['recall_one_training_after'] = recall_class_one
    SAVE_DATA['recall_zero_training_after'] = recall_class_zero

    # ==================== TEST ====================
    print('[INFOR] \n\n==================== TESTING STAGE ====================\n\n')

    NUM_IMG_TEST = FLAGS.num_img_test

    #get data test
    x_test, y_test = get_data_cifar(data_name='test_cifar10', num_img=NUM_IMG_TEST, class_name=CLASS_NAME)

    #preprocessing data
    x_test = pre_processing_data(model_name = MODEL_NAME, data = x_test)
    feature_x_test = model_k.predict(x_test)

    #project to a lower dimension
    pca = PCA(n_components=16, whiten=False)
    feature_x_test = pca.fit_transform(feature_x_test)

    #normalizate data
    feature_x_test = normalizate_min_max(feature_x_test, new_min=0, new_max=1)

    #get predicted_of_model
    predicted_of_model = get_predict(model, x_test, num_imgs=NUM_IMG_TEST)
    S = np.hstack((feature_x_test, predicted_of_model))

    # ===== P(x,1) BEFORE =====
    proba_one_test = []
    proba_two_before_test = []
    proba_two_after_test = []

    #normalization
    density_two_test = np.exp(P2O.score_samples(S))
    for density in zip(density_two_test):
        density_two_before_copy = copy.deepcopy(density_two_before)
        density_two_before_copy = np.append(density_two_before_copy, density)
        temp = density / np.sum(density_two_before_copy)
        proba_two_before_test.append(temp)

    # ===== P(x,1) AFTER =====
    proba_two_before_test = np.array(proba_two_before_test)
    proba_two_before_test = proba_two_before_test.reshape((proba_two_before_test.shape[0], -1))
    features_probas = np.hstack((feature_x_test, proba_two_before_test))
    features_probas = features_probas.dot(R)
    new_proba_two = features_probas[:,-1]

    #understanding as density
    new_density_two = new_proba_two.copy()
    for density in zip(new_density_two):
        density_two_after_copy = copy.deepcopy(density_two_after)
        density_two_after_copy = np.append(density_two_after_copy, density)
        temp = density / np.sum(density_two_after_copy)
        proba_two_after_test.append(temp)

    # ===== P(x,0) =====

    density_one_test = np.exp(P2Z.score_samples(S))
    for density in zip(density_one_test):
        density_one_copy = copy.deepcopy(density_one)
        density_one_copy = np.append(density_one_copy, density)
        temp = density / np.sum(density_one_copy)
        proba_one_test.append(temp)

    #get recall before transform
    recall_class_one, recall_class_zero = get_recall(proba_one_test, proba_two_before_test, NUM_IMG_TEST)
    SAVE_DATA['proba_one_testing'] = proba_one_test
    SAVE_DATA['proba_two_testing_before'] = proba_two_before_test

    print('[INFOR]: Recall testing of class {} before transform: {}'.format(CLASS_NAME, recall_class_one))
    print('[INFOR]: Recall testing of class not {} before transform: {}\n'.format(CLASS_NAME, recall_class_zero))

    SAVE_DATA['recall_one_testing_before'] = recall_class_one
    SAVE_DATA['recall_zero_testing_before'] = recall_class_zero

    #get recall after transform
    recall_class_one, recall_class_zero = get_recall(proba_one_test, proba_two_after_test, NUM_IMG_TEST)
    SAVE_DATA['proba_two_testing_after'] = proba_two_after_test

    print('[INFOR]: Recall testing of class {} after transform: {}'.format(CLASS_NAME, recall_class_one))
    print('[INFOR]: Recall testing of class not {} after transform: {}'.format(CLASS_NAME, recall_class_zero))

    SAVE_DATA['recall_one_testing_after'] = recall_class_one
    SAVE_DATA['recall_zero_testing_after'] = recall_class_zero

    SAVE_DATA = [SAVE_DATA]

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    np.save(os.path.join(FLAGS.output_dir, '{}_{}_{}.npy'.format(DATA_NAME, MODEL_NAME, CLASS_NAME)), SAVE_DATA)

    print('[INFOR]: Complete save data')


if __name__ == '__main__':

    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='ResNet50',
        help='Three options: ResNet50, EfficientNetB3, InceptionV3'
    )
    parser.add_argument(
        '--data_shape',
        type=int,
        nargs='+',
        help='Shape of data'
    )
    parser.add_argument(
        '--model_shape',
        type=int,
        nargs='+',
        help='Shape of model'
    )
    parser.add_argument(
        '--data_name',
        type=str,
        default='cifar10',
        help='''Name of dataset
        cifar10, fashion_mnist, skin_cancer
        '''
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default='airplane',
        help='''A class in the dataset
        cifar10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        fashion_mnist: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle-boot
        skin cancer: melanoma, nevus
        '''
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/npy_data',
        help='Output directory'
    )
    parser.add_argument(
        '--num_img_train',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_img_test',
        type=int,
        default=1000,
    )

    FLAGS = parser.parse_args()

    print("model_name = ", FLAGS.model_name)
    print("data_name = ", FLAGS.data_name)
    print("class_name = ", FLAGS.class_name)
    print("data_shape = ", FLAGS.data_shape)
    print("model_shape = ", FLAGS.model_shape)
    print("output_dir = ", FLAGS.output_dir)
    print("num_img_train = ", FLAGS.num_img_train)
    print("num_img_test = ", FLAGS.num_img_test)

    start_time = time.time()
    main(FLAGS)
    end_time = time.time()
    print('[INFOR] TOLTAL TIME: {}'.format(round((end_time-start_time), 4)))