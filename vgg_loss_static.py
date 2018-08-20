import time

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

from PIL import Image

import numpy as np

import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to content layers
    content_layers = ['block5_conv2']
    model_outputs = [vgg.get_layer(name).output for name in content_layers]
    # Build model
    return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))


def get_feature_representations(sess, model, image):

    features = sess.run(model.output, {model.input: image})

    return features

#tf.enable_eager_execution()
#print("Eager execution: {}".format(tf.executing_eagerly()))


image = load_and_process_img("img/school_bus.jpg")

with tf.Session() as sess:

    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    start_time = time.time()
    for _ in range(1000):
        features = get_feature_representations(sess, model, image)

    print('time: {:.4f}s'.format(time.time() - start_time))