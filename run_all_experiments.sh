#!/bin/bash
source ~/miniconda3/bin/activate cv_proj
# python run_experiment.py --spec experiments/simple01.json
# python run_experiment.py --spec experiments/simple01_extra.json
# python run_experiment.py --spec experiments/vgg01.json
# python run_experiment.py --spec experiments/vgg01_extra.json
# python run_experiment.py --spec experiments/vggpretrained01.json
# python run_experiment.py --spec experiments/vggpretrained01_extra.json

# Noise
python run_experiment.py --spec experiments/simple01_noise.json
python run_experiment.py --spec experiments/simple01_noise_extra.json
python run_experiment.py --spec experiments/vgg01_noise.json
python run_experiment.py --spec experiments/vgg01_noise_extra.json
python run_experiment.py --spec experiments/vggpretrained01_noise.json
python run_experiment.py --spec experiments/vggpretrained01_noise_extra.json
