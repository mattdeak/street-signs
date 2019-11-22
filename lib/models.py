import tensorflow as tf

# Aliasing instead of importing. This is just for dev
# since TF 2.0 has broken autocomplete basically.
# TODO: When finished, fix imports.
keras = tf.keras
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
EarlyStopping = tf.keras.callbacks.EarlyStopping


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


def build_simple_cnn1(
    input_shape,
    convolutional_blocks=2,
    convolutions_per_block=3,
    filters=64,
    lr=0.0001,
    dropout_rate=0.5,
    dense_units=128,
):
    input_layer = keras.layers.Input(shape=input_shape, name="Input")

    x = input_layer  # Alias for easier loops
    for block in range(convolutional_blocks):

        for layer in range(convolutions_per_block):
            x = Conv2D(filters, kernel_size, padding="valid", activation=tf.nn.selu)(x)
        x = tf.keras.layers.MaxPool2D()(x)

    flattened = Flatten()(x)
    dense1 = Dense(dense_units, activation="selu")(flattened)
    dropout = Dropout(dropout_rate)(dense1)
    outputs = Dense(N_OUTPUTS, activation="softmax")(dropout)

    model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])

    optimizer = tf.optimizers.Adam(lr=lr, amsgrad=True)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def build_vgg_model(input_shape, dense_units=512, activation="selu", lr=0.001):

    inputs = keras.layers.Input(shape=input_shape, name="Input")
    vgg_structure = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None)
    x = vgg_structure(inputs)
    x = Flatten()(x)
    x = Dense(512, activation="selu")(x)
    x = Dense(512, activation="selu")(x)
    outputs = Dense(N_OUTPUTS, activation="softmax")(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    # optimizer = tf.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    optimizer = tf.optimizers.Adam(lr=lr, clipnorm=1.0)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
