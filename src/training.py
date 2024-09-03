from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_preparation import data_gen


def compile_model(model, learning_rate=0.0001):
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def train_model(
    model, train_data_gen, val_data, val_labels, nb_epochs=3, nb_train_steps=100
):
    checkpoint = ModelCheckpoint(
        "best_model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001, verbose=1
    )
    history = model.fit(
        train_data_gen,
        epochs=nb_epochs,
        steps_per_epoch=nb_train_steps,
        validation_data=(val_data, val_labels),
        callbacks=[checkpoint, reduce_lr],
    )
    return history
