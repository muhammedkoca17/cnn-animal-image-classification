# src/train.py

import os
import warnings
import tensorflow as tf

from dataset import (
    RAW_PATH, TARGET_PATH, CLASSES,
    prepare_filtered_dataset,
    load_and_process_images,
    encode_and_split,
)
from augment import create_datagen, augment_dataset
from model import build_cnn
from visualizations import plot_history, plot_confusion

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.optimizer.set_experimental_options({"disable_xla": True})
warnings.filterwarnings("ignore")


def main():
    # 1) Dataset'i filtrele (Kaggle path ile aynı)
    prepare_filtered_dataset(RAW_PATH, TARGET_PATH, CLASSES, images_per_class=650)

    # 2) Görselleri yükle + normalize et
    X, y = load_and_process_images(TARGET_PATH, image_size=(128, 128))
    print(f"Dataset size: {len(X)} images")

    # 3) Label encode + train/test split
    X_train, X_test, y_train, y_test, encoder = encode_and_split(X, y)

    # 4) Augmentation
    datagen = create_datagen()
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, datagen, augment_per_image=2)

    # 5) Model
    num_classes = y_train.shape[1]   # one-hot shape ile bire bir
    model = build_cnn(input_shape=(128, 128, 3), num_classes=num_classes)

    # 6) Callbacks (notebook ile aynı)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # 7) Eğitim
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.2,
        epochs=40,
        batch_size=16,
        callbacks=[early_stopping, reduce_lr]
    )

    # 8) Orijinal test seti
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Original Test Accuracy: {test_accuracy * 100:.2f}%")

    # 9) Manipulated test set
    X_test_manipulated = manipulate_images(X_test)
    manipulated_loss, manipulated_accuracy = model.evaluate(X_test_manipulated, y_test)
    print(f"Manipulated Test Accuracy: {manipulated_accuracy * 100:.2f}%")

    # 10) Color constancy
    X_test_wb = get_wb_images(X_test_manipulated)
    wb_loss, wb_accuracy = model.evaluate(X_test_wb, y_test)
    print(f"Accuracy on Color-Corrected Test Images: {wb_accuracy * 100:.2f}%")

    # 11) Grafikleri çiz
    plot_history(history)

    # 12) Modeli kaydet
    os.makedirs("models", exist_ok=True)
    model.save("models/best_model.h5")


if __name__ == "__main__":
    main()
