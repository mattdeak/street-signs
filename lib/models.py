import tensorflow as tf
from utils import load_train, load_test
from pipelines import *

# Aliasing instead of importing. This is just for dev
# since TF 2.0 has broken autocomplete basically.
# TODO: When finished, fix imports.
keras = tf.keras
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
selu = tf.keras.activations.selu

# Baseline Model
images, labels = load_train()
val_images, val_labels = load_test()
dataset = make_train_pipeline(images, labels)
validation = make_validation_pipeline(val_images, val_labels)
input_shape = images.shape[1:]

# Hyperparameters
N_DENSE_UNITS = 128
N_OUTPUTS = 10
kernel_size = 3

input_layer = keras.layers.Input(shape=input_shape, name="Input")
model = selu(Conv2D(64, kernel_size, padding="same")(input_layer))
model = selu(Conv2D(64, kernel_size, padding="same")(model))
model = selu(Conv2D(64, kernel_size, padding="same")(model))
outputs = Dense(N_OUTPUTS, activation="softmax")(model)

model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit_generator(dataset,
