import tensorflow as tf
from lib.utils import load_train, load_test
from lib.pipelines import *
from lib.models import *
from lib.trainers import *
import pickle
from functools import partial
import os
import argparse
import json


def validate_spec(spec_data):
    assert spec_data["architecture"] in [
        "baseline",
        "simple",
        "vgg16",
    ], f"Architecture {spec_data['architecture']} not supported"


def main(builder, trainer, save_path, history_path, use_extra=False):

    # Hyperparameters

    images, labels = load_train(use_extra=use_extra)
    val_images, val_labels = load_test()

    steps_per_epoch = (
        len(images) // 32
    )  # We're going to validate more often than after an epoch
    validation_steps = len(val_images) // 32

    dataset = make_train_pipeline(images, labels)
    validation = make_validation_pipeline(val_images, val_labels)

    input_shape = images.shape[1:]

    model = builder(input_shape)
    history = trainer(
        model,
        dataset,
        validation,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    print(f"Saving trained model in {save_path}")
    tf.keras.models.save_model(model, save_path)

    print(f"Saving results in {history_path}")
    with open(history_path, "wb+") as results_file:
        pickle.dump(history.history, results_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default="/home/matt/Projects/CV/final/experiments/pretrained_01.json",
        required=False,
        help="Path to json file with experiment specification",
    )
    args = parser.parse_args()

    if not os.path.exists(args.spec):
        raise FileNotFoundError("Path to spec file not found")

    with open(args.spec, "r") as file:
        name = os.path.basename(args.spec)[:-5]
        spec_data = json.load(file)

    # Validate the json file
    validate_spec(spec_data)

    if spec_data["architecture"] == "simple":
        builder = partial(build_simple_cnn1, **spec_data["builder_args"])
        trainer = partial(train_from_scratch, **spec_data["trainer_args"])
    elif spec_data["architecture"] == "baseline":
        builder = partial(build_baseline)
        trainer = partial(train_from_scratch, **spec_data["trainer_args"])
    elif spec_data["architecture"] == "vgg16":
        builder = partial(build_vgg_model, **spec_data["builder_args"])
        if spec_data["builder_args"].get("weights") == "imagenet":
            trainer = partial(train_from_pretrained, **spec_data["trainer_args"])
        else:
            trainer = partial(train_from_scratch, **spec_data["trainer_args"])
    save_dir = "/home/matt/Projects/CV/final/saved_models/"
    results_dir = "/home/matt/Projects/CV/final/experiments/results/"

    save_path = os.path.join(save_dir, name)
    results_path = os.path.join(results_dir, name)

    main(builder, trainer, save_path, results_path)
