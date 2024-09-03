from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

data_dir = Path("data/chest_xray/")
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"


def load_train():
    normal_cases_dir = train_dir / "NORMAL"
    pneumonia_cases_dir = train_dir / "PNEUMONIA"
    normal_cases = normal_cases_dir.glob("*.jpeg")
    pneumonia_cases = pneumonia_cases_dir.glob("*.jpeg")
    train_data = []
    train_label = []
    for img in normal_cases:
        train_data.append(img)
        train_label.append("NORMAL")
    for img in pneumonia_cases:
        train_data.append(img)
        train_label.append("PNEUMONIA")
    df = pd.DataFrame(train_data)
    df.columns = ["images"]
    df["labels"] = train_label
    df = df.sample(frac=1).reset_index(drop=True)
    return df
