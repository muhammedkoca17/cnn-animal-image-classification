# src/visualizations.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FIG_DIR = "reports/figures"

os.makedirs(FIG_DIR, exist_ok=True)

def plot_history(history, save=True):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "training_curves.png")
        plt.savefig(path, dpi=200)
        print(f"[INFO] Training curves saved to {path}")
    plt.close()


def plot_confusion(y_true, y_pred, class_names, save=True):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)

    if save:
        path = os.path.join(FIG_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved to {path}")
    plt.close()
