## Running the Pipeline
#### Preparing for run.py
Models are located at: LINK. Please ensure that the models are unzipped in `saved_models`, otherwise they won't be loaded.

#### Running 
Simply run `python run.py`. This will automatically run the number recognition pipeline on all images in `input` and output them to `graded_images`.


## Running a Classification Experiment
### Preparing Spec

Specifications for an experiment are provided in `json` format in the `experiments` folder.
Important fields for designing an experiment are:
`architecture`: The architecture to use. Supports `baseline`, `simple` and `vgg16`
`builder_args`: This is where model hyperparameters are defined. Supported arguments depend on the architecture. Examine keyword arguments of functions in `models.py` for an exhaustive list.
`trainer_args`: This is where parameters related specifically to training are specified. Includes arguments like `max_epochs` and `patience` for early stopping.
`data`: All specifications must have at least one dataset. Supported datasets are `svhn_extra`, `svhn_noextra` and `format2_negative`.
`data_params`: This is where arguments related to the dataloader are specified. Can include preprocessors to use. Available preprocessors are in `lib/preprocess.py`, though only `add_noise` is currently supported as a field here.

Example specifications are already loaded in the experiments folder for reference.

### Running Experiment
When you have a spec defined, you can run it simply by using `python run_experiment.py --spec PATH_TO_SPEC`. Training curves will automatically be saved in experiments/results.

### Evaluating an Experiment
To evaluate an experiment on a test set after running, simply use `python evaluate_experiment.py --spec PATH_TO_SPEC --test-dataset DATASET_NAME`. If left blank, `test-dataset` will default to `svhn_noextra`.

### Models Used
Saved models are located in `saved_models`.
They map to the paper as:

simple01_noise_extra = 'Model 1'
vgg01_noise_extra = 'Model 2'
vggpretrained01_noise_extra= 'Model 3'

Their corresponding specifications can be found in the experiments folder.

