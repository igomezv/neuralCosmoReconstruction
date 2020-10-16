from NeuralNet import NeuralNet
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D, MaxPooling2D
from tensorflow.keras import Input

class ConvolAE(NeuralNet):
    def __init__(self, X, y, epochs=100, split=0.8, batch_size=64, min_delta=0,
                 patience=10, lr=0.0001):
        split = split
        self.batch_size = batch_size
        self.epochs = epochs
        min_delta = min_delta
        patience = patience
        self.lr = lr
        super(ConvolAE, self).__init__(X, y, epochs=epochs, split=split,
                                       batch_size=batch_size, min_delta=min_delta,
                                       patience=patience)

    def model(self):
        input_cov = Input(shape=(1048, 1048, 1))
        # x = layers.Conv2D(1, (50, 50), activation='relu', padding='same')(input_cov)
        # # shape_1 = [1048 - (n-1)] x [1048 - (n-1)]
        # x = layers.MaxPooling2D((3, 3))(x)
        # # shape_2 = [(shape_1[0] -1)/2]
        x = Conv2D(6, (3, 3), activation='relu', padding='same')(input_cov)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (4, 4), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((4, 4), padding='same')(x)

        x = Conv2D(4, (4, 4), activation='relu', padding='same')(encoded)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(6, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # x = layers.Flatten()(x)
        # encoded = layers.Dense(1048)(x)

        autoencoder = tf.keras.Model(input_cov, decoded)
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        autoencoder.summary()

        return autoencoder