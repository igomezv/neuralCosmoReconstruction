from NeuralNet import NeuralNet
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D, MaxPooling2D
from tensorflow.keras import Input

class ConvolAE(NeuralNet):
    def __init__(self, X, y, **kwargs):
        split = kwargs.pop('split', 0.8)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.epochs = kwargs.pop('epochs', 500)
        min_delta = kwargs.pop('min_delta', 0)
        patience = kwargs.pop('patience', 10)
        self.lr = kwargs.pop('lr', 0.0001)
        super(ConvolAE, self).__init__(X, y, **kwargs)

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
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.summary()

        return autoencoder