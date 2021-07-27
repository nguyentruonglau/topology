from keras.preprocessing.image import img_to_array, smart_resize
from keras.preprocessing.image import load_img
from imutils.paths import list_images
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def get_data(path_to_image):

  img_paths = list(list_images(path_to_image))
  print(len(img_paths))
  data = []

  for path in img_paths:
    temp = load_img(path); temp = img_to_array(temp)
    temp = smart_resize(temp, (192, 256))
    data.append(temp)
  
  data=np.array(data)
  print(data.shape)

  np.save('test_nevus.npy', data)


if __name__ == '__main__':
	path_to_image = './test/nv/'
	get_data(path_to_image)