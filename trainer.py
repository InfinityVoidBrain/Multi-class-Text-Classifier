# src/trainer.py
def train_model(model, train_padded, train_labels, val_padded, val_labels, epochs):
    history = model.fit(
        train_padded, train_labels,
        epochs=epochs,
        validation_data=(val_padded, val_labels),
        verbose=2
    )
    return history
