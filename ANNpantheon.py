from NeuralNet import NeuralNet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt

file = 'data/pantheon.txt'
data = pd.read_csv(file, sep=" ", usecols=['zcmb', 'zhel', 'mb', 'dmb'] )
print(data.head())

syscov = np.loadtxt('data/pantheon_errors.txt', skiprows=1).reshape((1048,1048))
print(syscov[:10, :])

x = data.values
r = np.random.uniform(0, np.max(x), size=100)
ALL = np.concatenate([x, syscov], axis=1)
# scalerx = StandardScaler()
# scalerx.fit(x.reshape(-1,1))
# X = scalerx.transform(x.reshape(-1,1))
# Xinv = scalerx.inverse_transform(X)

# neural_net for all
# neuralnetPantheon = NeuralNet(x, x)
neuralnetPantheon = NeuralNet(ALL, ALL)
neuralnetPantheon.plot(outputname='lossPantheonFull', show=True)

decoder = neuralnetPantheon.encoder()
a = decoder.predict(np.random.rand((1052,1)))
print(a)


# pred_mb = neuralnetMB.predict(scalerx.transform(r.reshape(-1,1)))
# plt.scatter(Xinv, mb, c='r', label='Real', alpha=0.2)
# plt.scatter(r, pred_mb, c='g', label='Fake', alpha=0.2)
# plt.savefig('MB.png')
# plt.show()

