import tensorflow.keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

class AutoEncoder:
    def __init__(self, X, y, **kwargs):
        try:
            _, self.input_nodes = np.shape(X)
        except:
            self.input_nodes = 1
        try:
            _, self.output_nodes = np.shape(y)
        except:
            self.output_nodes = 1
        self.initializer = RandomNormal()
        split = kwargs.pop('split', 0.8)
        self.first_nodes = kwargs.pop('first_nodes', 600)
        self.hidden_nodes = kwargs.pop('hidden_nodes', 300)
        self.last_nodes = kwargs.pop('last_nodes', 200)
        self.coded_nodes = kwargs.pop('coded_nodes', 50)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.epochs = kwargs.pop('epochs', 500)
        min_delta = kwargs.pop('min_delta', 0)
        patience = kwargs.pop('patience', 10)

        randomize = np.random.permutation(len(X))
        X = X[randomize]
        y = y[randomize]
        ntrain = int(split * len(X))
        indx = [ntrain]
        self.X_train, self.X_test = np.split(X, indx)
        self.y_train, self.y_test = np.split(y, indx)

        input_x = Input(shape=(self.input_nodes,))
        self.autoencoder = Model(input_x, self.autoencoder(input_x))
        self.autoencoder.compile(loss='mean_squared_error', optimizer="adam")
        self.autoencoder.summary()
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           min_delta=min_delta,
                                                           patience=patience,
                                                           restore_best_weights=True)]
        self.trained_model = self.fit()

    def autoencoder(self, input_x):
        # encoder
        efirst = Dense(self.first_nodes, activation='relu', input_shape=(self.input_nodes,))(input_x)
        ehidden = Dense(self.hidden_nodes, activation='relu')(efirst)
        elast = Dense(self.last_nodes, activation='relu')(ehidden)
        coded = Dense(self.coded_nodes, activation='relu')(elast)
        # decoder
        dfirst = Dense(self.last_nodes, activation='relu')(coded)
        dhidden = Dense(self.hidden_nodes, activation='relu')(dfirst)
        dlast = Dense(self.first_nodes, activation='relu')(dhidden)
        decoded = Dense(self.output_nodes, activation='relu')(dlast)

        return decoded

    def fit(self):
        return self.autoencoder.fit(self.X_train, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs, verbose=1,
                                    validation_data=(self.X_test, self.y_test),
                                    callbacks=self.callbacks)

    def predict(self, _X):
        return self.autoencoder.predict(_X)

    def plot(self, **kwargs):
        outputname = kwargs.pop('outputname', 'loss_AE')
        train_color = kwargs.pop('train_color', 'r')
        val_color = kwargs.pop('val_color', 'g')
        show = kwargs.pop('show', False)
        title = kwargs.pop('title', 'AE loss function')

        plt.plot(self.trained_model.history['loss'], color=train_color)
        plt.plot(self.trained_model.history['val_loss'], color=val_color)
        plt.title(title)
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('{}.png'.format(outputname))
        if show:
            plt.show()




