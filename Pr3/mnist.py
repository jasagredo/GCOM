from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Clasificadores import *
from PCA import *

def one_hot(x):
    can = np.zeros(10)
    can[int(x)] = 1
    return np.array(can)

def train_mnist():
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
    print("Comprimiendo conjunto de entrenamiento por PCA...")
    pca = PCA()
    train_x = pca.compresion(train_x)
    print("Comprimiendo conjunto de test por PCA...")
    test_x = pca.project(test_x)

    print("Entrenando clasificador LDA...")
    lda = LDA_classifier(train_x, train_t)
    lda.train(0.1)
    print("LDA preparado")
    return test_x, test_t, lda

def test_mnist(test_x, test_t, lda):
    cl = test_x.shape[1]
    print('Comienza el test')
    res = lda.classify(test_x)
    mal = len(filter((lambda x: x[0] != x[1]), zip(res, test_t)))
    print('Resultados')
    print('Mal clasificado: {0:.4f}'.format(mal*100/cl))


if __name__ == '__main__':
    test_x, test_t, perceptrones = train_mnist()
    test_mnist(test_x, test_t.flatten(), perceptrones)