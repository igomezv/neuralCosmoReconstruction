from NeuralNet import NeuralNet
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input

class ConvolAE(NeuralNet):
    def __init__(self, X, **kwargs):
        super(ConvolAE, self).__init__(self, X, y=X, **kwargs)

    def model(self):
        input_cov = Input(shape=(1048, 1048, 1))
        x = layers.Conv2D(10, (50, 50), activation='relu')(input_cov)
        # shape_1 = [1048 - (n-1)] x [1048 - (n-1)]
        x = layers.MaxPooling2D((4, 4))(x)
        # shape_2 = [(shape_1[0] -1)/2]
        x = layers.Flatten()(x)
        x = layers.Dense(1048)(x)
        # autoencoder.add(Dense(10))

        # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        # encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        autoencoder = tf.keras.Model(input_cov, x)
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        autoencoder.summary()