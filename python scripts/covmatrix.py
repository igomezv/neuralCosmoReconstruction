from ConvolAE import ConvolAE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import scipy as sp
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

file = 'data/pantheon_errors.txt'
# file = 'data/jla_v0_covmatrix.dat'
ndata = 1048
# ndata = 740
syscov = np.loadtxt(file, skiprows=1).reshape((ndata, ndata))

def checking_matrix(matrix):
    c_cplx = 0
    c_nosym = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if np.iscomplex(matrix[i, j]):
                c_cplx += 1
            if matrix[i, j] != matrix[j, i]:
                c_nosym += 1
                matrix[j, i] = matrix[i, j]
    print("Complex values: {} | No symmetrical: {}".format(c_cplx, c_nosym))
    return matrix


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


print("Covariance matrix")
syscov = checking_matrix(syscov)
view_matrix(syscov, rootname='original', imshow=False)

# For descomposition of symmetrical matrix syscov = P D P^T,
# where D is a diagonal matrix with the eigenvalues of syscov
# and P is an orthonormal matrix with the eigenvectors.

D, P = np.linalg.eigh(syscov)
D = np.diag(D)
# print("Eigenvector matrix:")
# checking_matrix(P)
# view_matrix(P, rootname='eigenvects', imshow=False)
recover_cov = P @ D @ P.T
view_matrix(recover_cov, 'recover_cov_first_pset', imshow=False)
# Generate a data set with matrices
# of eigenvectors from the original plus gaussian noise

numMatrix = 10
noise_factor = 1e-5
# scaler = StandardScaler()
# feature_range=(-1,1)
# scaler.fit(P)
pset = np.zeros((numMatrix, ndata, ndata))

# print("Generating dataset...")
for i in range(numMatrix):
    pset[i] = P + noise_factor * np.random.normal(loc=0.0, scale=0.01, size=P.shape)
    # pset[i] = scaler.transform(P) + noise_factor * np.random.normal(loc=0.0, scale=0.01, size=P.shape)
# print("Dataset generated!")

# print(np.shape(pset))
# view_matrix(pset[0, :, :], 'first_pset', imshow=False)
# try to recover original cov

# neural_net for mb
autoencoder = ConvolAE(pset, pset, batch_size=4, epochs=100, size=ndata)
autoencoder.plot(outputname='lossPantheon', imshow=False, show=True)

decoded_imgs = autoencoder.predict()
# view_matrix(decoded_imgs[0, :, :, 0], rootname='prediction_test', imshow=False)
new_matrix = decoded_imgs[0, :, :, 0] @ D @ np.transpose(decoded_imgs[0, :, :, 0])
view_matrix(new_matrix, rootname='new_cov', imshow=False)
