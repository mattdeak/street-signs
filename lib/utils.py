import sys
import numpy as np
import scipy

# Load format 2
FORMAT2_TRAIN = "/home/matt/Projects/CV/final/data/raw/format2/train_32x32.mat"
FORMAT2_TEST = "/home/matt/Projects/CV/final/data/raw/format2/test_32x32.mat"
FORMAT2_EXTRA = "/home/matt/Projects/CV/final/data/raw/format2/extra_32x32.mat"


def load_train(use_extra=False):
    if use_extra:
        print(
            "Warning - efficient handling of extra images not supported. This may cause memory issues"
        )
        cont = input("Continue? (y/n)")
        if cont.lower() != "y":
            print("Exiting")
            sys.exit(0)
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


images, labels = load_train(use_extra=True)
