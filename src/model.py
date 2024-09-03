from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Activation,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam


def create_custom_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64),
            Activation("relu"),
            Dense(2),
            Activation("softmax"),
        ]
    )
    return model


def vgg16_model(num_classes=2):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
