import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.models.model_mobilenet import build_mobilenet


TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/val"


BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def train_model():
    

    # -------------------------------
    # DATA GENERATORS
    # -------------------------------

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # -------------------------------
    # CLASS WEIGHTS
    # -------------------------------

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_data.classes
    )

    class_weight_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }

    print("Class Weights:", class_weight_dict)

    # -------------------------------
    # BUILD MODEL
    # -------------------------------

    model = build_mobilenet()

    # -------------------------------
    # STAGE 1 — TRAIN TOP LAYERS
    # -------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    checkpoint = ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.3,
        patience=2,
        verbose=1
    )

    print("\n===== STAGE 1 TRAINING =====\n")

    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # -------------------------------
    # STAGE 2 — FINE TUNING
    # -------------------------------

    print("\n===== STAGE 2 FINE TUNING =====\n")

    model.trainable = True

    for layer in model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
        # -------------------------------
    # TRAINING GRAPHS
    # -------------------------------

    import matplotlib.pyplot as plt

    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    auc = history1.history["auc"] + history2.history["auc"]
    val_auc = history1.history["val_auc"] + history2.history["val_auc"]

    epochs_range = range(1, len(acc) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("results/accuracy_plot.png")
    plt.show()

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("results/loss_plot.png")
    plt.show()

    # AUC Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, auc, label="Train AUC")
    plt.plot(epochs_range, val_auc, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training vs Validation AUC")
    plt.legend()
    plt.savefig("results/auc_plot.png")
    plt.show()


if __name__ == "__main__":
    train_model()