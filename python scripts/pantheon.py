from FFNet import FFNet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

file = 'data/pantheon.txt'
data = pd.read_csv(file, sep=" ", usecols=['zcmb', 'zhel', 'mb', 'dmb'] )
print(data.head())

syscov = np.loadtxt('data/pantheon_errors.txt', skiprows=1).reshape((1048,1048))
print(syscov[:10, :])

x = data.values[:, 0]
r = np.random.uniform(0, np.max(x), size=1000)
zhel = data.values[:, 1]
mb = data.values[:, 2]
dmb = data.values[:, 3]

scalerx = StandardScaler()
scalerx.fit(x.reshape(-1,1))
X = scalerx.transform(x.reshape(-1,1))
Xinv = scalerx.inverse_transform(X)

# neural_net for mb
neuralnetMB = FFNet(X, mb, [1, 100, 1])
neuralnetMB.plot(outputname='lossMB', show=True)
pred_mb = neuralnetMB.predict(scalerx.transform(r.reshape(-1,1)))
plt.scatter(Xinv, mb, c='r', label='Real', alpha=0.2)
plt.scatter(r, pred_mb, c='g', label='Fake', alpha=0.2)
plt.savefig('MB.png')
plt.show()

