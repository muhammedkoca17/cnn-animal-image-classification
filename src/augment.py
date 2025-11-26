import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_datagen():
    """
    Notebook'taki datagen = ImageDataGenerator(...) kısmının aynısı.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        vertical_flip=False,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen


def augment_dataset(X_train, y_train, datagen, augment_per_image: int = 2):
    """
    Notebook'taki augmented_images / augmented_labels döngüsünün aynısı.
    Her görüntü için augment_per_image kadar yeni örnek üretir.
    """
    augmented_images = []
    augmented_labels = []

    for i in range(len(X_train)):
        img = X_train[i]
        label = y_train[i]
        img = np.expand_dims(img, 0)

        count = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            count += 1
            if count >= augment_per_image:
                break

    X_train_aug = np.concatenate([X_train, np.array(augmented_images)])
    y_train_aug = np.concatenate([y_train, np.array(augmented_labels)])

    print("Increased training set size:", X_train_aug.shape)
    return X_train_aug, y_train_aug