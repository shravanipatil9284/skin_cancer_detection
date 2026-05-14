import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from src.models.model_mobilenet import build_mobilenet
from src.models.model_resnet import build_resnet
from src.models.model_efficientnet import build_efficientnet

TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/val"
TEST_DIR = "data/raw/test"

BATCH_SIZE = 16

# ONLY MODELS YOU TRAINED
models_dict = {
    "MobileNetV2": build_mobilenet,
    "ResNet50": build_resnet,
    "EfficientNetB0": build_efficientnet,
}

def get_data():
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(TRAIN_DIR, (224,224), batch_size=BATCH_SIZE, class_mode="binary")
    val_data = val_gen.flow_from_directory(VAL_DIR, (224,224), batch_size=BATCH_SIZE, class_mode="binary")
    test_data = test_gen.flow_from_directory(TEST_DIR, (224,224), batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)

    return train_data, val_data, test_data

def train_all():
    train_data, val_data, test_data = get_data()
    results = []

    for name in models_dict.keys():
        print(f"\n===== EVALUATING {name} =====\n")

        try:
            model_path = f"models/{name}.h5"

            # ✅ LOAD SAVED MODEL
            if os.path.isfile(model_path):
                print(f"✅ Loading saved model: {name}")
                model = tf.keras.models.load_model(model_path, compile=False)

                # ✅ Recompile (important)
                model.compile(
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

            else:
                print(f"⚠️ Model not found, skipping: {name}")
                continue

            # ✅ EVALUATION
            y_true = test_data.classes
            y_pred_prob = model.predict(test_data)
            y_pred = (y_pred_prob > 0.5).astype(int)

            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)

            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)

            # ✅ SAVE RESULTS
            results.append({
                "Model": name,
                "Accuracy": report["accuracy"],
                "Precision": report["1"]["precision"],
                "Recall": report["1"]["recall"],
                "F1-score": report["1"]["f1-score"],
                "AUC": roc_auc
            })

            # 🔥 SAVE AFTER EACH MODEL (important)
            pd.DataFrame(results).to_csv("models/model_results.csv", index=False)

            print(f"✅ {name} done\n")

        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")
            continue

    # FINAL OUTPUT
    if results:
        df = pd.DataFrame(results)
        print("\n===== FINAL COMPARISON =====")
        print(df)
    else:
        print("⚠️ No results generated")

if __name__ == "__main__":
    train_all()