# src/predict.py

import sys
import numpy as np
import cv2
import tensorflow as tf

from dataset import CLASSES   # aynı sınıf isimleri
from dataset import load_and_process_images   # istersen direkt burada basit preprocess de yazabiliriz

IMAGE_SIZE = (128, 128)
MODEL_PATH = "models/best_model.h5"


def preprocess_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path, model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    img = preprocess_single_image(image_path)
    preds = model.predict(img)[0]
    class_idx = np.argmax(preds)
    prob = float(preds[class_idx])
    class_name = CLASSES[class_idx]
    return class_name, prob


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    cls, p = predict_image(image_path)
    print(f"Predicted: {cls} ({p:.2f})")