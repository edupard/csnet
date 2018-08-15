import tensorflow as tf
import os

checkpoints_dir = './checkpoints/vgg'

checkpoint_file = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

def check_and_download_vgg_checkpoint():
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)

    from models.research.slim.datasets import dataset_utils

    url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)