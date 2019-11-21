import tensorflow as tf

# Aliasing instead of importing. This is just for dev
# since TF 2.0 has broken autocomplete basically.
# TODO: When finished, fix imports.
keras = tf.keras
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout


# Hyperparameters
N_DENSE_UNITS = 128
N_OUTPUTS = 10
kernel_size = 3

# Model Design
def build_baseline(input_shape):
    input_layer = keras.layers.Input(shape=input_shape, name="Input")
    conv1 = Conv2D(64, kernel_size, padding="same", activation=tf.nn.selu)(input_layer)
    conv2 = Conv2D(64, kernel_size, padding="same", activation=tf.nn.selu)(conv1)
    conv3 = Conv2D(64, kernel_size, padding="same", activation=tf.nn.selu)(conv2)
    flattened = Flatten()(conv3)
    dense1 = Dense(N_DENSE_UNITS, activation="selu")(flattened)
    dropout = Dropout(0.5)(dense1)
    outputs = Dense(N_OUTPUTS, activation="softmax")(dropout)

    model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])

    optimizer = tf.optimizers.Adam(lr=0.00001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
