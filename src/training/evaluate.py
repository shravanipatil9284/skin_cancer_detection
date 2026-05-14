import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
import seaborn as sns
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

TEST_DIR = "data/raw/test"
MODEL_PATH = "models/best_model.keras"

def evaluate_model():

    # Load model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")

    # Test data generator
    test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )

    # Predictions
    y_true = test_data.classes
    y_pred_prob = model.predict(test_data)
    y_pred = (y_pred_prob.flatten() > 0.35).astype(int)

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    print(f"✅ Test AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate_model()
