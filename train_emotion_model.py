import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.emotion_recognition.emotion_model import EmotionModel

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = os.path.join("data", "train")
TEST_DIR = os.path.join("data", "test")


def main():

    # Data augmentation + normalizare
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  #normalizare
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Construire model
    emotion_model = EmotionModel(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=5
    )

    model = emotion_model.model

    # Antrenare
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS
    )

    # Salvare model
    model.save("models/emotion_model.h5")

    print("Model salvat în models/emotion_model.h5")


if __name__ == "__main__":
    main()
