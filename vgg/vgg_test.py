import os
# tensorflow
import tensorflow as tf
import urllib2
slim = tf.contrib.slim
#vgg stuff
from models.research.slim.nets import vgg
from models.research.slim.datasets import imagenet
from models.research.slim.preprocessing import vgg_preprocessing

from vgg.download import check_and_download_vgg_checkpoint, checkpoint_file

# https://github.com/warmspringwinds/tensorflow_notes/blob/master/simple_classification_segmentation.ipynb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

check_and_download_vgg_checkpoint()

image_size = vgg.vgg_16.default_image_size

with tf.Graph().as_default():
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")

    # Open specified url and load image as a string
    image_string = urllib2.urlopen(url).read()

    # Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)

    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)

    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_file,
        slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        # Load weights
        init_fn(sess)

        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x: x[1])]





