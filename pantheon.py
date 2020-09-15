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
neuralnetMB = NeuralNet(X, mb)
neuralnetMB.plot(outputname='lossMB', show=True)
pred_mb = neuralnetMB.predict(scalerx.transform(r.reshape(-1,1)))
plt.scatter(Xinv, mb, c='r', label='Real', alpha=0.2)
plt.scatter(r, pred_mb, c='g', label='Fake', alpha=0.2)
plt.savefig('MB.png')
plt.show()

#neural net for zhel
neuralnetZHEL = NeuralNet(X, zhel)
neuralnetZHEL.plot(outputname='lossZHEL', show=True)
pred_zhel = neuralnetZHEL.predict(scalerx.transform(r.reshape(-1,1)))
plt.scatter(Xinv, zhel, c='r', label='Real', alpha=0.2)
plt.scatter(r, pred_zhel, c='g', label='Fake', alpha=0.2)
plt.savefig('ZHEL.png')
plt.show()

#neural net for dmb
neuralnetDMB = NeuralNet(X, dmb, first_nodes=200, hidden_nodes=200,
                             last_nodes=100, coded_nodes=2)
neuralnetDMB.plot(outputname='lossDMB', show=True)
pred_dmb = neuralnetDMB.predict(scalerx.transform(r.reshape(-1,1)))
plt.scatter(Xinv, dmb, c='r', label='Real', alpha=0.2)
plt.scatter(r, pred_dmb, c='g', label='Fake', alpha=0.2)
plt.savefig('DMB.png')
plt.show()

#neural net for cov
ALL = (Xinv.reshape(-1,1), zhel.reshape(-1,1), mb.reshape(-1,1), dmb.reshape(-1,1))
ALL = np.concatenate(ALL, axis=1)
neuralnetCOV = NeuralNet(ALL, syscov, first_nodes=800, hidden_nodes=500,
                             last_nodes=500, coded_nodes=100)
neuralnetCOV.plot(outputname='lossCOV', show=True)
ALLPRED = (r.reshape(-1,1), pred_zhel.reshape(-1,1),
           pred_mb.reshape(-1,1), pred_dmb.reshape(-1,1))
ALLPRED = np.concatenate(ALLPRED, axis=1)
pred_cov = neuralnetCOV.predict(ALLPRED)
print(pred_cov[:10, :])