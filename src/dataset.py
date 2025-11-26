import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Notebook'taki ile aynı sınıflar
CLASSES = [
    "collie", "dolphin", "elephant", "fox", "moose",
    "rabbit", "sheep", "squirrel", "giant+panda", "polar+bear"
]

# Kaggle'daki path'lerin bire bir karşılığı
RAW_PATH = "/kaggle/input/animals-with-attributes-2/Animals_with_Attributes2/JPEGImages"
TARGET_PATH = "/kaggle/working/FilteredImages"


def prepare_filtered_dataset(
    path: str = RAW_PATH,
    target: str = TARGET_PATH,
    classes=CLASSES,
    images_per_class: int = 650,
):
    """
    Notebook'taki filtreleme kodunun fonksiyon haline getirilmiş versiyonu.
    Kaggle'da bir kere çalıştırman yeterli.
    """
    os.makedirs(target, exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(path, class_name)
        target_path = os.path.join(target, class_name)

        if not os.path.exists(class_path):
            print(f"Source class path {class_path} does not exist. Skipping...")
            continue

        os.makedirs(target_path, exist_ok=True)

        print(f"Processing class: {class_name}")
        image_count = 0
        available_files = os.listdir(class_path)

        total_images = len(available_files)
        images_to_copy = min(images_per_class, total_images)
        print(f"Found {total_images} images. Attempting to copy {images_to_copy} images.")

        for file_name in available_files:
            if image_count >= images_to_copy:
                break

            full_file_name = os.path.join(class_path, file_name)
            if os.path.isfile(full_file_name):
                img = cv2.imread(full_file_name)
                if img is not None:
                    cv2.imwrite(os.path.join(target_path, file_name), img)
                    image_count += 1
                else:
                    print(f"Warning: Unable to read image {full_file_name}")

        print(f"Completed {image_count}/{images_to_copy} images for class {class_name}")


def load_and_process_images(data_dir: str, image_size=(128, 128)):
    """
    Notebook'taki load_and_process_images fonksiyonunun aynısı.
    target klasöründeki görüntüleri okur, 128x128 yapar, 0–1 normalize eder.
    """
    images = []
    labels = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        img_resized = cv2.resize(img, image_size)
                        img_normalized = img_resized / 255.0
                        images.append(img_normalized)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return np.array(images), np.array(labels)


def encode_and_split(X, y, test_size=0.3, random_state=42):
    """
    Notebook'taki:
      encoder = LabelEncoder()
      y_encoded = encoder.fit_transform(...)
      y_categorical = to_categorical(...)
      train_test_split(...)
    kısmının fonksiyonlaştırılmış hali.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, random_state=random_state
    )

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, encoder