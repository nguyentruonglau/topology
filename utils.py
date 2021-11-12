import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from database.index import get_name_from_index


def kde_plot(P, label):
    """Plot 2D PDF

    Args:
        P (pdf): probability density function
        S (2D): [N,M]

    Returns:
        [None]
    """
    #define domain value to plot
    print('[INFOR]: Plot PDF of {}'.format(label))
    num_point = 4000
    x_plot = np.linspace(0, 1, num_point)
    x_plot = x_plot.reshape(x_plot.shape[0], 1)
    x_plot = [list(item) for item in x_plot]
    x_plot = [item*33 for item in x_plot]

    #calculate the density value
    log_dens = np.array(P.score_samples(x_plot))

    fig, ax = plt.subplots()
  
    ax.plot(np.exp(log_dens), lw=2,
              linestyle='-', label='PDF of {}'.format(label))

    x_plot = np.linspace(0, 1, num_point)
    ax.fill(x_plot, np.exp(log_dens), facecolor='blue', alpha=0.8)

    ax.legend(loc='upper left')
    # plt.show()
    plt.savefig('./output/pdf_of_{}.jpg'.format(label))
    plt.close()
    return 0


def get_predict(model, x, num_imgs):
    """Create vector u, for the purpose of concancate with feature vetors
       u -> prediction of the model through the prediction process

    Args:
        model (model object): pretrained model
        x (2D array): data for prediction
        num_imgs (int): number of images

    Returns:
        [2D array]: predict vector
    """

    y_score = model.predict(x, verbose=1)
    ranks = np.argmax(y_score, axis=1)
    mask = np.argmax(y_score, axis=1)
    
    #the most mispredicted class of 1000 imagenet classes
    votes = np.zeros(1000)
    for it in ranks[0:num_imgs]:
        votes[it] += 1

    most_vote = np.argmax(votes)
    print("[INFOR]: Index of most vote   :{}".format(most_vote))
    print("[INFOR]: Class of most vote   :{}".format(get_name_from_index(most_vote)))
    print("[INFOR]: Number of most vote  :{}\n".format(votes[most_vote]))
    
    u = np.zeros(len(mask))
    u[mask==most_vote]=1
    u = u.reshape((u.shape[0], 1))

    votes = np.delete(votes, most_vote)
    second_most_vote = np.argmax(votes)
    print("\n[INFOR]: Index of second most vote     :{}".format(second_most_vote))
    print("[INFOR]: Class of second most vote     :{}".format(get_name_from_index(second_most_vote)))
    print("[INFOR]: Number of second most vote    :{}".format(votes[second_most_vote]))
    return u


def second_largest(ls_number):
    """Get second largest number

    Args:
      ls_number (list): list of numbers
    Returns:
      [int]: exist: second largest number 
             not exist: None
    """
    minimum = float('-inf')
    first, second = minimum, minimum
    for n in numbers:
        if n > first:
            first, second = n, first
        elif first > n > second:
            second = n
    return second if second != minimum else None


def get_model(model_name, data_shape, model_shape):
    """Get model architecture from model name

    Args:
        model_name (string): name of model
        data_shape (tuple): shape of data
        model_shape (tuple): shape of model

    Returns:
        [Model object]: model
    """
    print('[INFOR]: get pretrained model')
    input_tensor = tf.keras.Input(shape=data_shape)
    resized_images = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, model_shape))(input_tensor)

    if model_name == 'EfficientNetB3':
        model = tf.keras.applications.EfficientNetB3(
            include_top=True,
            weights='imagenet',
            input_tensor=resized_images,
            input_shape=(300, 300, 3),
            pooling='avg'
            )
    elif model_name == 'InceptionV3':
        model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights='imagenet',
            input_tensor=resized_images,
            input_shape=(299, 299, 3),
            pooling='avg'
            )
    elif model_name == 'ResNet50':
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights='imagenet',
            input_tensor=resized_images,
            input_shape=(224, 224, 3),
            pooling='avg'
            )
    else: raise ValueError('[ERROR]: not found pretrained model')

    return model


def pre_processing_data(model_name, data):
    """Preprocessing data

    Args:
        data (4D array): data
        model_name (string): name of model

    Returns:
        [4D array]: data after preprocessing
    """
    if model_name == 'EfficientNetB3': data = tf.keras.applications.efficientnet.preprocess_input(data)
    elif model_name == 'ResNet50': data = tf.keras.applications.resnet50.preprocess_input(data)
    elif model_name == 'InceptionV3': data = tf.keras.applications.inception_v3.preprocess_input(data)
    else: raise ValueError('[ERROR]: not found preprocessing')

    return data


def get_recall(proba_one, proba_two, num_img):
    """Get recall

    Args:
        proba_one, proba_two (1D array):

    Returns:
        [int]: recall
    """
    y_pred = [];
    for i,j in zip(proba_one, proba_two):
        if j >= i: y_pred.append(1)
        else: y_pred.append(0)

    y_pred_one = y_pred[0:num_img]
    y_pred_zero = y_pred[num_img:]

    recall_class_one = np.sum(y_pred_one)
    recall_class_zero = num_img - np.sum(y_pred_zero)
    return recall_class_one, recall_class_zero


def model_from_layer(model, name_layer):
    """Get new model with output is conv k_th of model

    Args:
        model (Model  object): model
        name_layer (string): name of layer

    Returns:
        [Model object]: model with output is conv k_th of model
    """
    #get all layer name
    layers = model.layers
    idx = 0
    for i, layer in enumerate(layers):
        if layer.name == name_layer:
            idx = i
            
    #get new model
    model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[idx].output)
    
    return model
