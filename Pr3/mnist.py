from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Clasificadores import *
from PCA import *

def one_hot(x):
    can = np.zeros(10)
    can[int(x)] = 1
    return can

def train_mnist():
    mnist = fetch_mldata('MNIST original', data_home = '~/Documents/Universidad/GCOM/Pr3')
    data = mnist.data
    target = mnist.target
    mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
    np.random.shuffle(mis_digitos)
    ochenta = int(np.rint(mis_digitos.shape[0]*0.8))
    train = mis_digitos[:ochenta, :]
    train_x = train[:, :train.shape[1]-1]
    train_t = train[:, train.shape[1]-1:]
    train_t = np.array(map(one_hot, train_t))
    test = mis_digitos[ochenta:, :]
    test_x = test[:, :test.shape[1]-1]
    test_t = test[:, test.shape[1]-1:]
    test_t = np.array(map(one_hot, test_t))

    lda = LDA_classifier(train_x.T, train_t.T)
    lda.train(0.3)
    return test_x, test_t, lda

def test_mnist(test_x, test_t, lda):
    mal = 0
    cl = 0
    print(test_t)
    print('Comienza el test')
    for i in range(test_x.shape[0]):
        res = lda.classify(test_x[i])
        cl += 1
        if res != test_t[i]:
            mal += 1
    print('Resultados')
    print('Mal clasificado: {0:.4f}'.format(mal*100/cl))




if __name__ == '__main__':
    test_x, test_t, perceptrones = train_mnist()
    test_mnist(test_x, test_t.flatten(), perceptrones)