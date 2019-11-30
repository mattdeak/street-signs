import tensorflow as tf
import argparse
from tqdm import tqdm
import os
import scipy.io
import numpy as np
import cv2

class IllegalImageShapeError(Exception):
    pass


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    # Else..
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_mat(matfile):
    data = scipy.io.loadmat(matfile)
    images = data["X"]
    labels = data["y"]
    labels = labels.reshape(-1,)
    images = np.moveaxis(images, -1, 0)
    return images, labels


def load_png(filepath):
    """Retrieves image from png"""
    image = cv2.imread(filepath)
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_npy(image_filepath, label_filepath=None):
    """Retrieves image from npy file"""
    images = np.load(image_filepath)
    images = np.moveaxis(images, -1, 0)

    if label_filepath:
        labels = np.load(label_filepath)
        labels = labels.reshape(-1,)
        return images, labels

    return images


def make_example(image, label):
    h, w, _ = image.shape

    encoded_image = cv2.imencode(".jpg", image)[1].tostring()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": _int64_feature(h),
                "image/width": _int64_feature(w),
                "image/image": _bytes_feature(encoded_image),
                "image/label": _int64_feature(label),
            }
        )
    )
    return example


def write_records(data, output_path):
    writer = tf.io.TFRecordWriter(output_path)

    for image, label in data:
        example = make_example(image, label)
        writer.write(example.SerializeToString())

    writer.close()


def make_shvn_format2(src_dir, target_dir, validation_split=0.2, use_extra=False):

    train_filepath = os.path.join(src_dir, "train_32x32.mat")
    test_filepath = os.path.join(src_dir, "test_32x32.mat")
    extra_images_filepath = os.path.join(src_dir, "extra_images_32x32.npy")
    extra_labels_filepath = os.path.join(src_dir, "extra_labels_32x32.npy")

    target_train_path = os.path.join(target_dir, "train.tfrecord")
    target_val_path = os.path.join(target_dir, "val.tfrecord")
    target_test_path = os.path.join(target_dir, "test.tfrecord")

    images, labels = load_mat(train_filepath)


    if use_extra:
        print("Loading Extra Files")
        extra_images, extra_labels = load_npy(
            extra_images_filepath, extra_labels_filepath
        )
        # Shuffle these into the images
        images = np.vstack([images, extra_images])
        labels = np.hstack([labels, extra_labels])
        shuffle_idx = np.random.permutation(np.arange(len(images)))
        images = images[shuffle_idx]
        labels = labels[shuffle_idx]

    # Fix labels
    labels = labels - 1

    # Generate Splits
    N = images.shape[0]
    train_size = int(N * (1 - validation_split))
    idx = np.random.permutation(np.arange(N))
    train_idx = idx[:train_size]
    val_idx = idx[train_size:]

    print("Splitting train into train/val")
    train_images = images[train_idx]
    train_labels = labels[train_idx]

    val_images = images[val_idx]
    val_labels = labels[val_idx]

    print("Writing Train")
    write_records(zip(train_images, train_labels), target_train_path)

    print("Writing Val")
    write_records(zip(val_images, val_labels), target_val_path)

    test_images, test_labels = load_mat(test_filepath)
    test_labels = test_labels - 1 # Fix indexing
    print("Writing Test")
    write_records(zip(test_images, test_labels), target_test_path)


def make_format2_negatives(src_dir, target_dir, validation_split=0.2):

    train_dir = os.path.join(src_dir, "train")
    test_dir = os.path.join(src_dir, "test")
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")]

    assert len(files) > 0, "No png files in designated directory"

    target_train = os.path.join(target_dir, "train.tfrecord")
    target_val = os.path.join(target_dir, "val.tfrecord")
    target_test = os.path.join(target_dir, "test.tfrecord")

    # Shuffle them up
    np.random.shuffle(files)
    train_size = int(len(files) * (1 - validation_split))

    print("Writing Train Records")

    train_writer = tf.io.TFRecordWriter(target_train)
    for img_file in tqdm(files[:train_size]):
        try:
            image = preprocess_format2_neg(load_png(img_file))
        except IllegalImageShapeError:
            continue


        # All negatives are labelled 10, because
        # We have 10 legitimate classes so max positive index is 9.
        example = make_example(image, 10)
        train_writer.write(example.SerializeToString())

    train_writer.close()

    print("Writing Val Records")
    val_writer = tf.io.TFRecordWriter(target_val)
    for img_file in tqdm(files[train_size:]):
        try:
            image = preprocess_format2_neg(load_png(img_file))
        except IllegalImageShapeError:
            continue

        example = make_example(image, 10)
        val_writer.write(example.SerializeToString())

    val_writer.close()

    print("Writing Test Records")
    test_writer = tf.io.TFRecordWriter(target_test)
    for img_file in tqdm(test_files):
        try:
            image = preprocess_format2_neg(load_png(img_file))
        except IllegalImageShapeError:
            continue

        example = make_example(image, 10)
        test_writer.write(example.SerializeToString())

    test_writer.close()

def preprocess_format2_neg(image):
    h, w, _ = image.shape
    if h < 32 or w < 32:
        raise IllegalImageShapeError('Image dimensions are too small')

    return image[:32, :32]


def rewrite_svhn(validation_split=0.2):
    print("Writing records to extra and noextra")
    src_directory = "data/raw/format2/"
    target_noextra_dir = "data/svhn/noextra"
    target_extra_dir = "data/svhn/extra"

    print("-------- Beginning NoExtra Write -------")
    make_shvn_format2(
        src_directory, target_noextra_dir, validation_split=validation_split
    )
    print("-------- Beginning Extra Write -------")
    make_shvn_format2(
        src_directory,
        target_extra_dir,
        validation_split=validation_split,
        use_extra=True,
    )


def rewrite_format1_negatives(validation_split=0.2):
    src_directory = "data/raw/format1"
    target_dir = "data/format2_negatives"

    make_format2_negatives(src_directory, target_dir, validation_split=validation_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['svhn','format1_negative'])
    parser.add_argument('--validation_size', default=0.2, required=False)
    args = parser.parse_args()

    if args.dataset == 'svhn':
        rewrite_svhn(validation_split=args.validation_size)
    elif args.dataset == 'format1_negative':
        rewrite_format1_negatives(validation_split=args.validation_size)
