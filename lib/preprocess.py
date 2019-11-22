import tensorflow as tf
import os
import cv2
import numpy as np
import scipy.io
from abc import abstractmethod, ABC


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, X, y):
        """The preprocessing function to be applied to a tf.Dataset"""


class OneHotLabels(Preprocessor):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, image, labels):
        ohe_labels = tf.one_hot(labels, self.n_classes)
        return image, ohe_labels


class ResizeImage(Preprocessor):
    def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR):
        self.size = size
        self.method = method

    def __call__(self, image, labels):
        """Resizes an image"""
        resized = tf.image.resize_images(image, (self.size), method=self.method)
        return resized, labels


class FloatifyImage(Preprocessor):
    def __call__(self, image, labels):
        """Converts an image from 0-255 int to 0-1 float format."""
        return image / 255, labels


class RandomAugmentation(Preprocessor):
    def __init__(
        self,
        vertical_flip=True,
        horizontal_flip=True,
        contrast_deltas=None,
        brightness_maxdelta=None,
    ):
        self.vertical = vertical_flip
        self.horizontal = horizontal_flip
        self.contrast_deltas = contrast_deltas
        self.brightness_delta = brightness_maxdelta

    def __call__(self, image, labels):
        if self.vertical:
            image = tf.image.random_flip_up_down(image)

        if self.horizontal:
            image = tf.image.random_flip_left_right(image)

        if self.brightness_delta:
            image = tf.image.random_brightness(image, self.brightness_delta)
            image = tf.clip_by_value(
                image, 0.0, 1.0
            )  # In case brightness scaling goes too high or low

        if self.contrast_deltas:
            image = tf.image.random_contrast(
                image, self.contrast_deltas[0], self.contrast_deltas[1]
            )
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, labels


class Maybe(Preprocessor):
    def __init__(self, preprocessor, chance=0.5):
        """A preprecessing super-layer that gives another preprocessor a
        predefined chance of actually being executed.
        
        Arguments:
            preprocessor {Preprocessor} -- The preprocessor
        
        Keyword Arguments:
            chance {float} -- The chance of a preprocessor being executed (default: {0.5})
        """
        assert isinstance(
            preprocessor, Preprocessor
        ), "Preprocessor must be a Preprocessing Function"
        assert (
            chance < 1.0 and chance > 0.0
        ), "Chance should be between 0 and 1 exclusive"
        self.preprocessor = preprocessor
        self.chance = chance

    def __call__(self, image, labels):
        r = tf.random.uniform([1])[0]

        cond = tf.cond(
            r < self.chance,
            true_fn=lambda: self.preprocessor(image, labels),
            false_fn=lambda: (image, labels),
        )
        return cond


class DropLabels(Preprocessor):
    def __call__(self, image, labels):
        return image
