import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class NeuralNet(object):
    def __init__(self, X, y, **kwargs):

        split = kwargs.pop('split', 0.8)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.epochs = kwargs.pop('epochs', 500)
        min_delta = kwargs.pop('min_delta', 0)
        patience = kwargs.pop('patience', 10)

        randomize = np.random.permutation(len(X))
        X = X[randomize]
        y = y[randomize]
        ntrain = int(split * len(X))
        indx = [ntrain]
        self.X_train, self.X_test = np.split(X, indx,  axis=0)
        self.y_train, self.y_test = np.split(y, indx,  axis=0)
        self.model = self.model()

        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           min_delta=min_delta,
                                                           patience=patience,
                                                           restore_best_weights=True)]
        self.trained_model = self.fit()

    def model(self, **kwargs):
        err = "NeuralNet: You need to implement an model"
        raise NotImplementedError(err)

    def fit(self):
        return self.model.fit(self.X_train, self.y_train,
                              batch_size=self.batch_size,
                              epochs=self.epochs, verbose=1,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=self.callbacks)

    def predict(self, _X):
        return self.model.predict(_X)

    def plot(self, **kwargs):
        outputname = kwargs.pop('outputname', 'loss')
        train_color = kwargs.pop('train_color', 'r')
        val_color = kwargs.pop('val_color', 'g')
        show = kwargs.pop('show', False)
        title = kwargs.pop('title', 'Loss function')

        plt.plot(self.trained_model.history['loss'], color=train_color)
        plt.plot(self.trained_model.history['val_loss'], color=val_color)
        plt.title(title)
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('{}.png'.format(outputname))
        if show:
            plt.show()




