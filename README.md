# neuralCosmoReconstruction

In the directory **notebooks** there are the neural networks used for
H(z) and fs(8) data. Also the hyper-parameters tunning using
tensorboard. Other scripts in this folder are several 
attemps for manage the covariance matrix with autoencoders.

In the root directory:

- *NeuralNet.py*: General base class for neural network

- *FFNet.py*: Feed forward neural network (child of NeuralNet)

- *ConvolAE.py*: Convolutional AutoEncoder (child of NeuralNet)

- *pantheon.py*: Use of FFNet in pantheon data (z, mb, dmb)

- *covmatrix.py*: Use of ConvolAE for the systematic errors 
covariance matrix for pantheon observations.