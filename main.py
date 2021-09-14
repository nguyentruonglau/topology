from geometric import get_matrix_geometric_transformation
from geometric import calculate_total_norm
from generate.utils import get_data
from utils import kde_plot, get_model
from utils import pre_processing_data
from kernel_density import get_pdf_positive
from kernel_density import get_pdf_negative
from geometric import normalizate_density

from utils import model_from_layer
from utils import get_recall
from utils import get_predict
from sklearn.decomposition import PCA
from geometric import normalizate_min_max

import os
import numpy as np
import argparse
import copy
import time

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def main(FLAGS):
    start_train_time = time.time()
    #save data to excel file
    output_file = os.path.join(FLAGS.output_dir, '{}_{}.csv'.format(FLAGS.dataset_name.lower(), 
        FLAGS.model_name.lower())
        )
    file = open(output_file, 'a')
    save_data = ''

    print('==================== TRAINING STAGE ====================')
    num_img_train = int(FLAGS.class_name_train.split('_')[-1])
    save_data += FLAGS.class_name_train.split('_')[0] + ','#index
    if int(FLAGS.class_name_train.split('_')[0]) == 0:
        file.write('index, total_norm_before, time_r, total_norm_after,\
        sen_train_before, spe_train_before, sen_train_after, spe_train_after, time_train,\
        sen_test_before, spe_test_before, sen_test_after, spe_test_after, time_test \n')
    
    #get data train
    x_train, y_train = get_data(FLAGS.dataset_name, 'train', FLAGS.class_name_train)
    
    #get model
    model_shape = tuple(FLAGS.model_shape)
    model = get_model(FLAGS.model_name, FLAGS.data_shape, model_shape)

    #preprocessing data
    x_train = pre_processing_data(FLAGS.model_name, data = x_train)

    #get predicted_of_model
    predicted_of_model = get_predict(model, x_train, num_img_train)

    #get last layer
    model_k = model_from_layer(model, name_layer='avg_pool')

    #estimate pdf - positive
    F, S1O, S2O, P1O, P2O = get_pdf_positive(model_k, num_img_train, x_train, y_train, predicted_of_model)

    # ===== P(x,1) BEFORE =====
    density_two_before = np.exp(P2O.score_samples(S2O))
    proba_two_before = normalizate_density(density_two_before)

    #calculate total norm
    total_norm = calculate_total_norm(F, P1O, P2O, S1O, S2O, FLAG='before_transform')
    print('[INFOR]: Toltal norm before transform: {:.4f}'.format(total_norm))
    save_data += str(round(total_norm, 4)) + ','#total_norm_before

    #get matrix R
    start_r_time = time.time()
    R = get_matrix_geometric_transformation(F, S1O, S2O, P1O, P2O)
    end_r_time = time.time()
    save_data += str(round((end_r_time - start_r_time), 4)) + ','#time_r

    # ===== P(x,1) AFTER =====
    total_norm, density_two_after = calculate_total_norm(F, P1O, P2O, S1O, S2O, R=R, FLAG='after_transform')
    proba_two_after = normalizate_density(density_two_after)
    print('[INFOR]: Toltal norm after transform: {:.4f}'.format(total_norm))
    save_data += str(round(total_norm, 4)) + ','#total_norm_after

    #estimate pdf - negative
    _, _, S2Z, _, P2Z = get_pdf_negative(model_k, num_img_train, x_train, y_train, predicted_of_model)

    # ===== P(x,0) =====
    density_one = np.exp(P2Z.score_samples(S2Z))
    proba_one = normalizate_density(density_one)

    #get recall before transform
    recall_class_one, recall_class_zero = get_recall(proba_one, proba_two_before, num_img_train)

    print('[INFOR]: SEN training of class {} before transform: {}'.format(FLAGS.class_name_train, recall_class_one))
    print('[INFOR]: SPE training of class {} before transform: {}\n'.format(FLAGS.class_name_train, recall_class_zero))
    save_data += str(round(recall_class_one/num_img_train, 4)*100) + ','#sen_train_before
    save_data += str(round(recall_class_zero/num_img_train, 4)*100) + ','#spe_train_before

    #get recall after transform
    recall_class_one, recall_class_zero = get_recall(proba_one, proba_two_after, num_img_train)

    print('[INFOR]: SEN training of class {} after transform: {}'.format(FLAGS.class_name_train, recall_class_one))
    print('[INFOR]: SPE training of class {} after transform: {}'.format(FLAGS.class_name_train, recall_class_zero))
    save_data += str(round(recall_class_one/num_img_train, 4)*100) + ','#sen_train_after
    save_data += str(round(recall_class_zero/num_img_train, 4)*100) + ','#spe_train_after

    end_train_time = time.time()
    save_data += str(round((end_train_time - start_train_time), 0)) + ','#time_train

    print('[INFOR] \n\n==================== TESTING STAGE ====================\n\n')

    start_test_time = time.time()
    num_img_test = int(FLAGS.class_name_test.split('_')[-1])

    #get data test
    x_test, y_test = get_data(FLAGS.dataset_name, 'test', FLAGS.class_name_test)

    #preprocessing data
    x_test = pre_processing_data(FLAGS.model_name, data = x_test)
    feature_x_test = model_k.predict(x_test)

    #project to a lower dimension
    pca = PCA(n_components=16, whiten=False)
    feature_x_test = pca.fit_transform(feature_x_test)

    #normalizate data
    feature_x_test = normalizate_min_max(feature_x_test, new_min=0, new_max=1)

    #get predicted_of_model
    predicted_of_model = get_predict(model, x_test, num_imgs=num_img_test)
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
    recall_class_one, recall_class_zero = get_recall(proba_one_test, proba_two_before_test, num_img_test)

    print('[INFOR]: SEN testing of class {} before transform: {}'.format(FLAGS.class_name_test, recall_class_one))
    print('[INFOR]: SPE testing of class {} before transform: {}\n'.format(FLAGS.class_name_test, recall_class_zero))
    save_data += str(round(recall_class_one/num_img_test, 4)*100) + ','#sen_test_before
    save_data += str(round(recall_class_zero/num_img_test, 4)*100) + ','#spe_test_before

    #get recall after transform
    recall_class_one, recall_class_zero = get_recall(proba_one_test, proba_two_after_test, num_img_test)

    print('[INFOR]: SEN testing of class {} after transform: {}'.format(FLAGS.class_name_test, recall_class_one))
    print('[INFOR]: SPE testing of class {} after transform: {}'.format(FLAGS.class_name_test, recall_class_zero))
    save_data += str(round(recall_class_one/num_img_test, 4)*100) + ','#sen_test_after
    save_data += str(round(recall_class_zero/num_img_test, 4)*100) + ','#spe_test_after

    print('[INFOR]: Complete save data')
    end_test_time = time.time()
    save_data += str(round((end_test_time - start_test_time), 4))#time_test
    file.write(save_data); file.write("\n")
    file.close()


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
        type=tuple,
        default=(32,32,3),
        help='Shape of data'
    )
    parser.add_argument(
        '--model_shape',
        type=int,
        nargs='+',
        help='Shape of model'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='cifar_100',
        help='''Name of dataset
        cifar-10, cifar_100, fgvc_aircraft, stanford_car
        '''
    )
    parser.add_argument(
        '--class_name_train',
        type=str,
        default='0_500',
        help='''A class in the dataset: index_number of images
        '''
    )
    parser.add_argument(
        '--class_name_test',
        type=str,
        default='0_100',
        help='''A class in the dataset: index_number of images
        '''
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/csv_data',
        help='''Use for save data
        '''
    )

    FLAGS = parser.parse_args()

    print("model_name = ", FLAGS.model_name)
    print("dataset_name = ", FLAGS.dataset_name)
    print("class_name_train = ", FLAGS.class_name_train)
    print("class_name_test = ", FLAGS.class_name_test)
    print("data_shape = ", FLAGS.data_shape)
    print("model_shape = ", FLAGS.model_shape)
    print("output_dir = ", FLAGS.output_dir)

    main(FLAGS)