import os
import cv2
import numpy as np
import itertools
import logging
from lib import preprocess
import tensorflow as tf

# Data Locations

logging.basicConfig()

FORMAT1_TRAIN = "data/raw/format1/train"
FORMAT1_TEST = "data/raw/format1/test"

DIRECTORIES = {
    "svhn_noextra": "data/svhn/noextra/",
    "svhn_extra": "data/svhn/extra/",
    "format2_negative": "data/format2_negatives/",
}

N_OUTPUTS = 10


class DataLoader:
    def __init__(
        self, dataset_spec, num_epochs=100, batch_size=16, shuffle_buffer_len=10000
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle_buffer_len = shuffle_buffer_len

        if any("negative" in name for name in dataset_spec):
            self.has_negative_class = True
        else:
            self.has_negative_class = False

        self.n_classes = N_OUTPUTS
        if self.has_negative_class:
            self.n_classes += 1

        train_datasets = []
        val_datasets = []
        test_datasets = []

        self.train_size = 0
        self.val_size = 0

        for dataset_name in dataset_spec:
            assert dataset_name in DIRECTORIES, f"{dataset_name} not supported"
            train_subset = self.load_train_dataset(dataset_name)
            val_subset = self.load_val_dataset(dataset_name)

            # We have to do this incrementally here, as the interleave
            # operation disrupts the count
            self.train_size += train_subset.reduce(0, lambda x, _: x + 1).numpy()
            self.val_size += val_subset.reduce(0, lambda x, _: x + 1).numpy()

            # Repeating ensures that there are no end-of-sequences reached through interleaving
            # Although this is pretty inelegant. There must be a better way
            train_datasets.append(train_subset.repeat())
            val_datasets.append(val_subset.repeat())

        # Interleave each list of datasets into 1
        train_dataset = self.interleave(train_datasets)
        val_dataset = self.interleave(val_datasets)

        self.train_dataset = (
            train_dataset.repeat(self.num_epochs)
            .shuffle(self.shuffle_buffer_len)
            .batch(self.batch_size)
            .prefetch(1)
        )
        self.val_dataset = val_dataset.batch(self.batch_size).prefetch(1)

    def load_train_dataset(self, dataset_name):
        root_dir = DIRECTORIES[dataset_name]
        train_record = os.path.join(root_dir, "train.tfrecord")
        return self._load_dataset(train_record)

    def load_val_dataset(self, dataset_name):
        root_dir = DIRECTORIES[dataset_name]
        val_record = os.path.join(root_dir, "val.tfrecord")
        return self._load_dataset(val_record)

    def generate_test_dataset(self, dataset_name):
        root_dir = DIRECTORIES[dataset_name]
        test_record = os.path.join(root_dir, "test.tfrecord")
        test_set = self._load_dataset(test_record)
        test_set = test_set.batch(self.batch_size).prefetch(1)
        return test_set

    def _load_dataset(self, record_path):
        dataset = tf.data.TFRecordDataset(record_path)
        dataset = dataset.map(preprocess.parse_example)
        # Preprocessors
        dataset = dataset.map(preprocess.OneHotLabels(self.n_classes))
        return dataset

    def interleave(self, datasets):
        """interleave

        Combines an array of datasets into one dataset via interleaving

        Parameters
        ----------

        datasets :

        Returns
        -------
        """
        combined_dataset = datasets[0]
        for i in range(1, len(datasets)):
            combined_dataset = tf.data.Dataset.zip(
                (combined_dataset, datasets[i])
            ).flat_map(
                lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
                    tf.data.Dataset.from_tensors(x1)
                )
            )
        return combined_dataset

