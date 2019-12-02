import argparse
import json
from lib.dataloader import DataLoader
import tensorflow as tf
import os

SAVED_MODEL_DIR = 'saved_models'

class UnsavedModelError(Exception):
    pass

def parse_spec():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec')
    parser.add_argument('--test-dataset', default='svhn_noextra')
    args = parser.parse_args()
    return args.spec, args.test_dataset

if __name__ == "__main__":
    spec, dataset = parse_spec()

    if not os.path.exists(spec):
        raise FileNotFoundError("Path to spec file not found")

    with open(spec, "r") as file:
        name = os.path.basename(spec)[:-5]
        spec_data = json.load(file)
    

    model_path = os.path.join(SAVED_MODEL_DIR, f'{name}.hp5')
    if not os.path.exists(model_path):
        raise UnsavedModelError("Run the experiment before evaluating")

    datasets = spec_data['data']['datasets']
    data_params = spec_data['data'].get('data_params', {})


    loader = DataLoader(datasets, **data_params)
    test_data = loader.generate_test_dataset(dataset)

    # TODO: This could easily be done to exhaustion. Alter DataLoader API to handle test sets better
    

    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(test_data) 

    print(f"Model Results: {results}")


