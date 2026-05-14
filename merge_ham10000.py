import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
BASE_DIR = r"D:\B.Tech\FY project\skin-cancer-detect\data"
HAM_DIR = os.path.join(BASE_DIR, "ham10000_raw")
IMAGE_DIR = os.path.join(HAM_DIR, "images")
CSV_PATH = os.path.join(HAM_DIR, "GroundTruth.csv")

TARGET_DIR = os.path.join(BASE_DIR, "raw")

# ---------------- CLEAN OLD DATA ----------------
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

# Create folders
for split in ["train", "val", "test"]:
    for cls in ["benign", "malignant"]:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH)

# ---------------- ONE-HOT → CLASS ----------------
class_columns = ["MEL", "BCC", "AKIEC", "NV", "BKL", "DF", "VASC"]

def get_class(row):
    for col in class_columns:
        if row[col] == 1.0:
            return col
    return None

df["lesion_class"] = df.apply(get_class, axis=1)

# Drop invalid rows
df = df.dropna(subset=["lesion_class"])

# ---------------- REMOVE DUPLICATES ----------------
df = df.drop_duplicates(subset=["image"])

# ---------------- MAP TO BENIGN / MALIGNANT ----------------
malignant_classes = ["MEL", "BCC", "AKIEC"]

df["final_class"] = df["lesion_class"].apply(
    lambda x: "malignant" if x in malignant_classes else "benign"
)

# ---------------- ADD IMAGE FILENAME ----------------
df["filename"] = df["image"] + ".jpg"

# ---------------- STRATIFIED SPLIT ----------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["final_class"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["final_class"],
    random_state=42
)

# ---------------- COPY FILES ----------------
def copy_images(dataframe, split):
    count = 0
    for _, row in dataframe.iterrows():
        src = os.path.join(IMAGE_DIR, row["filename"])
        dst = os.path.join(TARGET_DIR, split, row["final_class"], row["filename"])

        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1

    print(f"{split.upper()} copied: {count}")

copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("\n✅ CLEAN DATASET CREATED (NO LEAKAGE)")