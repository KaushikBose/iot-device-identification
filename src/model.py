"""CNN model definition."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)


def build_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 3e-4,
    pooling: str = "avg",
) -> tf.keras.Model:
    """Build and compile the spectrogram CNN with configurable pooling."""
    if pooling not in {"avg", "max", "avgmax", "flatten"}:
        raise ValueError(f"Unknown pooling type: {pooling}")

    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.003)(inputs)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    if pooling == "flatten":
        x = Flatten()(x)
    elif pooling == "avg":
        x = GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = GlobalMaxPooling2D()(x)
    else:
        x = Concatenate()([GlobalAveragePooling2D()(x), GlobalMaxPooling2D()(x)])

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
