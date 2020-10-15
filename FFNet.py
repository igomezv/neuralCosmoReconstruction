import tensorflow.keras as K
from NeuralNet import NeuralNet


class FFNet(NeuralNet):
    def __init__(self, X, y, topology, **kwargs):
        self.topology = topology
        split = kwargs.pop('split', 0.8)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.epochs = kwargs.pop('epochs', 500)
        min_delta = kwargs.pop('min_delta', 0)
        patience = kwargs.pop('patience', 10)
        self.lr =  kwargs.pop('lr', 0.0001)
        super(FFNet, self).__init__(X, y, **kwargs)

    def model(self):
        # encoder
        model = K.models.Sequential()
        # Hidden layers
        for i, nodes in enumerate(self.topology):
            if i == 0:
                model.add(K.layers.Dense(self.topology[1], input_dim=self.topology[0], activation='relu'))
            elif i < len(self.topology) - 2:
                model.add(K.layers.Dense(self.topology[i + 1], activation='relu'))
            else:
                # Last layer (output)
                model.add(K.layers.Dense(self.topology[i], activation='relu'))
        optimizer = K.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        model.summary()
        return model

