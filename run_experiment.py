import tensorflow as tf
from lib.utils import load_train, load_test
from lib.models import *
from lib.trainers import *
from lib.dataloader import DataLoader
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


def main(builder, trainer, loader, save_path, history_path, use_extra=False):

    # Hyperparameters
    input_shape = (32, 32, 3)
    classes = loader.n_classes


    train_set = loader.train_dataset
    val_set = loader.val_dataset

    steps_per_epoch = loader.train_size // loader.batch_size  
    validation_steps = loader.val_size // loader.batch_size

    model = builder(input_shape, n_classes=classes)
    history = trainer(
        model,
        train_set,
        val_set,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    print(f"Saving trained model in {save_path}")
    tf.keras.models.save_model(model, save_path, save_format='h5')
    print(f"Saving results in {history_path}")
    with open(history_path, "wb+") as results_file:
        pickle.dump(history.history, results_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default="/home/matt/Projects/CV/final/experiments/simple01.json",
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

    datasets = spec_data['data']['datasets']

    data_params = spec_data['data'].get('data_params', {})
    loader = DataLoader(datasets, **data_params)

    save_path = os.path.join(save_dir, f'{name}.hp5')
    results_path = os.path.join(results_dir, name)

    main(builder, trainer, loader, save_path, results_path)
