import os
import shutil
import random
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)

def create_folders(base_path):
    for folder in ["train", "val", "test"]:
        for cls in ["benign", "malignant"]:
            os.makedirs(os.path.join(base_path, folder, cls), exist_ok=True)

def process_and_save(img_path, out_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img.save(out_path)

def prepare_dataset(raw_dir, out_dir, test_size=0.2, val_size=0.1):
    images = []
    labels = []

    for label in ["benign", "malignant"]:
        class_path = os.path.join(raw_dir, label)
        for img_name in os.listdir(class_path):
            images.append(os.path.join(class_path, img_name))
            labels.append(label)

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size)

    create_folders(out_dir)

    def save_split(X, y, split_name):
        for img, label in zip(X, y):
            out_path = os.path.join(out_dir, split_name, label, os.path.basename(img))
            process_and_save(img, out_path)

    save_split(X_train, y_train, "train")
    save_split(X_val, y_val, "val")
    save_split(X_test, y_test, "test")

if __name__ == "__main__":
    prepare_dataset("data/raw", "data/processed")
