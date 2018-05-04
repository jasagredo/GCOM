from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Clasificadores import *
from PCA import *

def one_hot(x):
    can = np.zeros(10)
    can[int(x)] = 1
    return np.array(can)

def obtenerMNIST():
    print("Cargando MNIST...")
    mnist = fetch_mldata('MNIST original', data_home = '~/Documents/Universidad/GCOM/Pr3')
    data = mnist.data
    target = mnist.target
    print("Mezclando datos...")
    mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
    np.random.shuffle(mis_digitos)
    ochenta = int(np.rint(mis_digitos.shape[0]*0.8))
    train = mis_digitos[:ochenta, :]
    train_x = train[:, :train.shape[1]-1].T
    train_t = train[:, train.shape[1]-1:]
    train_t = np.vstack(map(one_hot, train_t)).T
    test = mis_digitos[ochenta:, :]
    test_x = test[:, :test.shape[1]-1].T
    test_t = test[:, test.shape[1]-1:]
    print("Comprimiendo datos por PCA...")
    pca = PCA()
    data = pca.compresion(np.hstack([train_x, test_x]), tol=0.0001)
    train_x = data[:, :ochenta]
    test_x = data[:, ochenta:]
    print("\n")
    return test_t, test_x, train_t, train_x


def mnist_lda(test_t, test_x, train_t, train_x):
    print("Entrenando clasificador LDA...")
    lda = LDA_classifier(train_x, train_t)
    lda.train(0.0001)
    print("LDA preparado")
    test_t = test_t.flatten()
    cl = test_x.shape[1]
    print('Comienza el test...')
    res = lda.classify(test_x)
    mal = len(filter((lambda x: x[0] != x[1]), zip(res, test_t)))
    print('#### Resultados:')
    print('Mal clasificado: {0:.4f}\n'.format(mal * 100 / cl))

def mnist_ls(test_t, test_x, train_t, train_x):
    ls = LeastSquares()
    print("Entrenando Least Squares...")
    ls.train(train_x, train_t)
    print("Least Squares preparado")
    test_t = test_t.flatten()
    cl = test_x.shape[1]
    print("Comienza el test...")
    res = ls.classify(test_x)
    mal = len(filter((lambda x: x[0] != x[1]), zip(res, test_t)))
    print('#### Resultados:')
    print('Mal clasificado: {0:.4f}'.format(mal * 100 / cl))

if __name__ == '__main__':
    test_t, test_x, train_t, train_x = obtenerMNIST()
    mnist_lda(test_t, test_x, train_t, train_x)
    mnist_ls(test_t, test_x, train_t, train_x)
