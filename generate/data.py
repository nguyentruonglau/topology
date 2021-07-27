import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def generate_binary_data_cifar10(output_dir, num_img):
    """
    Generate binary data from cifar10 dataset, 
    this object or not this object.
    All are randomly selected.

    Args:
      num_img (int): number of images to use

    Returns:
        [None]
    """
    #define
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    NUM_CLASSES = len(classes)

    #download data
    (_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
    NUM_EACH_CLASS = 1000

    #sort data
    y = y.reshape((y.shape[0], ))
    airplane = x[y==0]; automobile = x[y==1]; bird = x[y==2]
    cat = x[y==3]; deer = x[y==4]; dog = x[y==5]
    frog = x[y==6]; horse = x[y==7]; ship = x[y==8]
    truck = x[y==9]

    #update x
    x = np.vstack((airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck))

    for i in range(NUM_CLASSES):
      #get data label one
      class_one = x[NUM_EACH_CLASS*i : NUM_EACH_CLASS*i+NUM_EACH_CLASS]
      np.random.shuffle(class_one)
      class_one = class_one[0:num_img]
      print('{} - SHAPE: {}'.format(classes[i], class_one.shape))
      print(class_one[0][0][0])

      #get data label zero, not one
      class_zero = np.vstack((x[0:NUM_EACH_CLASS*i], x[NUM_EACH_CLASS*i+NUM_EACH_CLASS:50000]))
      np.random.shuffle(class_zero)
      class_zero = class_zero[0:num_img]
      print('NOT {} - SHAPE: {}'.format(classes[i], class_zero.shape))

      #data
      data = np.vstack((class_one, class_zero))
      print('SHAPE OF DATA {}: {}'.format(classes[i], data.shape))
      print(data[0][0][0])

      #check output directory
      if not os.path.exists('./data/{}'.format(output_dir)): os.mkdir('./data/{}'.format(output_dir))

      #check images
      fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
      ax[0].imshow(data[0])
      ax[0].set_title('{}'.format(classes[i]))
      ax[0].set_xticks([])
      ax[0].set_yticks([])

      ax[1].imshow(data[num_img])
      ax[1].set_title('NOT {}, EXAMPLE'.format(classes[i]))
      ax[1].set_xticks([])
      ax[1].set_yticks([])
      plt.show()

      #save data
      np.save('./data/{}/{}.npy'.format(output_dir, classes[i]), data)
      print()
    print()



def generate_binary_data_fashion_mnist(output_dir, num_img):
    """
    Generate binary data from fashion mnist dataset, 
    this object or not this object.
    All are randomly selected.

    Args:
      num_img (int): number of images to use

    Returns:
        [None]
    """
    #define
    classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal','shirt', 'sneaker', 'bag', 'ankle-boot']
    NUM_CLASSES = len(classes)

    #download data
    (_, _), (x, y) = tf.keras.datasets.fashion_mnist.load_data()
    x = np.array(x)
    NUM_EACH_CLASS = 1000

    #sort data
    y = y.reshape((y.shape[0], ))
    t_shirt = x[y==0]; trouser = x[y==1]; pullover = x[y==2]
    dress = x[y==3]; coat = x[y==4]; sandal = x[y==5]
    shirt = x[y==6]; sneaker = x[y==7]; bag = x[y==8]
    ankle_boot = x[y==9]

    #update x
    x = np.vstack((t_shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle_boot))

    for i in range(NUM_CLASSES):
      #get data label one
      class_one = x[NUM_EACH_CLASS*i:NUM_EACH_CLASS*i+NUM_EACH_CLASS]
      np.random.shuffle(class_one)
      class_one = class_one[0:num_img]
      #expand for class one
      class_one = tf.expand_dims(class_one, axis=-1)
      class_one = tf.repeat(class_one, repeats=3, axis=-1)
      class_one = tf.cast(class_one, dtype=tf.float32)
      print('{} - SHAPE: {}'.format(classes[i], class_one.shape))

      #get data label zero, not one
      class_zero = np.vstack((x[0:NUM_EACH_CLASS*i], x[NUM_EACH_CLASS*i+NUM_EACH_CLASS:60000]))
      np.random.shuffle(class_zero)
      class_zero = class_zero[0:num_img]
      #expand for class zero
      class_zero = tf.expand_dims(class_zero, axis=-1)
      class_zero = tf.repeat(class_zero, repeats=3, axis=-1)
      class_zero = tf.cast(class_zero, dtype=tf.float32)
      print('NOT {} - SHAPE: {}'.format(classes[i], class_zero.shape))

      #data
      data = np.vstack((class_one, class_zero))
      print('SHAPE OF DATA {}: {}'.format(classes[i], data.shape))

      #check images
      fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
      ax[0].imshow(data[0])
      ax[0].set_title('{}'.format(classes[i]))
      ax[0].set_xticks([])
      ax[0].set_yticks([])

      ax[1].imshow(data[num_img])
      ax[1].set_title('NOT {}, EXAMPLE'.format(classes[i]))
      ax[1].set_xticks([])
      ax[1].set_yticks([])
      plt.show()

      #check output directory
      if not os.path.exists('./data/{}'.format(output_dir)): os.mkdir('./data/{}'.format(output_dir))

      #save data
      np.save('./data/{}/{}.npy'.format(output_dir, classes[i]), data)
      print()
    print()