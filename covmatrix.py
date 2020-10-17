from ConvolAE import ConvolAE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import scipy as sp
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

file = 'data/pantheon_errors.txt'
syscov = np.loadtxt(file, skiprows=1).reshape((1048,1048))


def checking_matrix(matrix):
    c_cplx = 0
    c_nosym = 0
    for i in range(len(matrix)):
        for j in range(len(syscov)):
            if np.iscomplex(syscov[i, j]):
                c_cplx += 1
            if syscov[i, j] != syscov[j, i]:
                c_nosym += 1
                syscov[j, i] = syscov[i, j]
    # print("Complex values: {} | No symmetrical: {}".format(c_cplx, c_nosym))
    return True


def view_matrix(matrix, rootname="test", imshow=True, heatmap=True, show=True):
    if imshow:
        plt.imshow(matrix)
        plt.savefig("{}_imshow.png".format(rootname))
        if show:
            plt.show()
    if heatmap:
        sns.heatmap(pd.DataFrame(matrix), annot=False, fmt='g', xticklabels=False,
                    yticklabels=False,
                    cmap='inferno')
        plt.savefig("{}_heatmap.png".format(rootname))
        if show:
            plt.show()
    return True


# print("Covariance matrix")
# checking_matrix(syscov)
# view_matrix(syscov, rootname='original', imshow=False)

# For descomposition of symmetrical matrix syscov = P D P^T,
# where D is a diagonal matrix with the eigenvalues of syscov
# and P is an orthonormal matrix with the eigenvectors.

D, P = sp.linalg.eigh(syscov)
# print("Eigenvector matrix:")
# checking_matrix(P)
# view_matrix(P, rootname='eigenvects', imshow=False)

# Generate a data set with matrices
# of eigenvectors from the original plus gaussian noise

numMatrix = 10
noise_factor = 1e-5
# scaler = StandardScaler()
# feature_range=(-1,1)
# scaler.fit(ortM)
pset = np.zeros((numMatrix, 1048, 1048))

# print("Generating dataset...")
for i in range(numMatrix):
    pset[i] = P + noise_factor * np.random.normal(loc=0.0, scale=0.01, size=P.shape)
# eigenvecdata[i] = scaler.transform(eigenvecdata[i])
# print("Dataset generated!")

print(np.shape(pset))
# view_matrix(pset[0, :, :], 'first_pset', imshow=False)

# neural_net for mb
autoencoder = ConvolAE(pset, pset, batch_size=64, epochs=50)
autoencoder.plot(outputname='lossPantheon', imshow=False, show=True)

decoded_imgs = autoencoder.predict()
view_matrix(decoded_imgs[0, :, :, 0], rootname='prediction_test', imshow=False)