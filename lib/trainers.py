import tensorflow as tf
from .models import *


def train_from_scratch(
    model,
    train_dataset,
    val_dataset,
    max_epochs=100,
    steps_per_epoch=32,
    validation_steps=32,
    patience=3
):
    """train_from_scratch

    Parameters
    ----------

    input_shape : Shape of data
    train_dataset : tf.Data dataset used for training
    val_dataset : tf.Data dataset used for validation

    Returns
    -------
    history
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def train_from_pretrained(
    model,
    train_dataset,
    val_dataset,
    max_epochs=100,
    steps_per_epoch=32,
    validation_steps=32,
    n_frozen_epochs=0,
    patience=3
):
    """train_from_pretrained

    Parameters
    ----------

    model : Keras model
    train_dataset : tf.Data dataset used for training
    val_dataset : tf.Data dataset used for validation
    max_epochs : maximum number of epochs to run
    steps_per_epoch : number of batches per epoch
    validation_steps : number of batches per epoch of validation set
    n_frozen_epochs : number of epochs to run while freezing pretrained layers

    Returns
    -------
    """
    try:
        vgg_layer = model.get_layer("vgg16")
    except:
        raise ValueError(
            "train_from_pretrained requires the model to incorproate a vgg16 architecture"
        )

    # Create early stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    ]

    # Freeze vgg layer for n_frozen_epochs
    # This can be useful for ensuring the gradient rolling off the dense layers
    # isn't too big
    if n_frozen_epochs > 0:
        print(f"Training with vgg frozen for {n_frozen_epochs} epochs")
        vgg_layer.trainable = False
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=n_frozen_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

    vgg_layer.trainable = True
    print("Training for a maximum of {max_epochs} epochs. Using early stopping")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    return history
