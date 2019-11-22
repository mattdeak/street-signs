import logging
import numpy as np
import scipy.io

# Load format 2
FORMAT2_TRAIN = "/home/matt/Projects/CV/final/data/raw/format2/train_32x32.mat"
FORMAT2_TEST = "/home/matt/Projects/CV/final/data/raw/format2/test_32x32.mat"
FORMAT2_EXTRA_IMAGES = (
    "/home/matt/Projects/CV/final/data/raw/format2/extra_images_32x32.npy"
)
FORMAT2_EXTRA_LABELS = (
    "/home/matt/Projects/CV/final/data/raw/format2/extra_labels_32x32.npy"
)


def load_train(use_extra=False):
    # Loading and merging with train
    train = scipy.io.loadmat(FORMAT2_TRAIN)
    images = train["X"]
    labels = train["y"]
    labels = labels.reshape(-1,)
    images = np.moveaxis(images, -1, 0)

    if use_extra:
        extra_images = np.load(FORMAT2_EXTRA_IMAGES)
        extra_labels = np.load(FORMAT2_EXTRA_LABELS)
        extra_labels = extra_labels.reshape(-1,)
        extra_images = np.moveaxis(extra_images, -1, 0)
        images = np.vstack([images, extra_images])
        labels = np.vstack([labels, extra_labels])
        # Randomly shuffle them together
        idx = np.random.permutation(np.arange(images.shape[0]))
        images = images[idx, :, :, :]
        labels = labels[idx]

    return images / 255.0, labels - 1  # Make range 0-9


def load_test():
    """load_test
    Loads the format2 test matrix into memory.
    Returns: images, labels
    """
    test = scipy.io.loadmat(FORMAT2_TEST)
    images = test["X"]
    labels = test["y"]
    labels = labels.reshape(-1,)
    images = np.moveaxis(images, -1, 0)

    return images / 255.0, labels - 1
