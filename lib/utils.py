import numpy as np
import scipy.io

# Load format 2
FORMAT2_TRAIN = "/home/matt/Projects/CV/final/data/raw/format2/train_32x32.mat"
FORMAT2_TEST = "/home/matt/Projects/CV/final/data/raw/format2/test_32x32.mat"
FORMAT2_EXTRA = "/home/matt/Projects/CV/final/data/raw/format2/extra_32x32.mat"


def load_train(use_extra=False):
    # Loading and merging with train
    train = scipy.io.loadmat(FORMAT2_TRAIN)
    images = train["X"]
    labels = train["y"]
    images = np.moveaxis(images, -1, 0)

    if use_extra:
        extra = scipy.io.loadmat(FORMAT2_EXTRA)
        extra_images = extra["X"]
        extra_labels = extra["y"]
        extra_images = np.moveaxis(extra_images, -1, 0)
        images = np.vstack([images, extra_images])
        labels = np.vstack([labels, extra_labels])

    return images, labels


def load_test():
    """load_test
    Loads the format2 test matrix into memory.
    Returns: images, labels
    """
    test = scipy.io.loadmat(FORMAT2_TEST)
    images = test["X"]
    labels = test["y"]
    images = np.moveaxis(images, -1, 0)

    return images, labels
