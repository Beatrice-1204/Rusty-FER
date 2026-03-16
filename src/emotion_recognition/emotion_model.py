import tensorflow as tf
from tensorflow.keras import layers, models


class EmotionModel:

   # Model CNN - pentru clasificare emoții

    def __init__(self, input_shape=(48, 48, 1), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    def _build_model(self):

        model = models.Sequential()  #straturile puse secvential


        # Bloc 1

        model.add(layers.Conv2D(32, (3, 3), padding="same",input_shape=self.input_shape))  #32 filtre - tipuri de caracteristici
        model.add(layers.BatchNormalization())  #stabilizeaza distribuita activarilor
        model.add(layers.Activation("relu"))    #non-litearitate
        model.add(layers.MaxPooling2D((2, 2)))   #reduce dimensiunea spatiala


        # Bloc 2

        model.add(layers.Conv2D(64, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D((2, 2)))


        # Bloc 3

        model.add(layers.Conv2D(128, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D((2, 2)))


        # Flatten

        model.add(layers.Flatten())  #transforma in 1D


        # Fully Connected

        model.add(layers.Dense(256))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))  #dezactivare 50% din neuroni


        # Output

        model.add(layers.Dense(self.num_classes, activation="softmax")) #trans din scoruri brute in probabilitati


        # Compile

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",  #masoara predictiile de realitate
            metrics=["accuracy"]
        )

        return model
