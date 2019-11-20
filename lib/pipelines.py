import tensorflow as tf
import scipy
import preprocess
import numpy as np
import cv2
from collections import namedtuple

# Just for debugging. TODO: Move this into a config file
FlagTuple = namedtuple("Flags", "shuffle_buffer_size batch_size num_epochs")
FLAGS = FlagTuple(shuffle_buffer_size=10000, batch_size=32, num_epochs=10)
# 10 Data directories for development
train = "/home/matt/Projects/CV/final/data/raw/format2/train_32x32.mat"


# TODO: Create composable pipeline.
floatify = preprocess.FloatifyImage()
onehot = preprocess.OneHotLabels(10)
augmentation = preprocess.RandomAugmentation(
    vertical_flip=False,
    horizontal_flip=False,
    contrast_deltas=None,
    brightness_maxdelta=0.3,
)

maybe_augment = preprocess.Maybe(augmentation)

f = scipy.io.loadmat(train)
images = f["X"]
labels = f["y"]
images = np.moveaxis(images, -1, 0)


def make_train_pipeline(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.map(floatify, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(onehot, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        maybe_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    dataset = dataset.repeat(FLAGS.num_epochs)
    return dataset


def make_validation_pipeline(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.map(floatify, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(onehot, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    dataset = dataset.repeat()  # Repeat this one forever
    return dataset


# Don't even know if we need this
def make_test_pipeline(images):
    pass
