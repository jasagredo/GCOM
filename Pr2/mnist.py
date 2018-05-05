from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Perceptron import *

def train_mnist():
    mnist = fetch_mldata('MNIST original', data_home = '~/Documents/Universidad/GCOM/Pr2')
    data = mnist.data
    target = mnist.target
    mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
    np.random.shuffle(mis_digitos)
    ochenta = int(np.rint(mis_digitos.shape[0]*0.8))
    train = mis_digitos[:ochenta, :]
    train_x = train[:, :train.shape[1]-1].T
    train_t = train[:, train.shape[1]-1:]
    test = mis_digitos[ochenta:, :]
    test_x = test[:, :test.shape[1]-1].T
    test_t = test[:, test.shape[1]-1:]
    perceptrones = []
    for i in range(10):
        print("Preparando datos para perceptron {0}".format(i))
        per = Perceptron(data.shape[1], 3)
        aux = np.equal(train_t, np.ones_like(train_t)*i).flatten()
        X_1 = train_x[:, aux]
        X_2 = train_x[:, np.logical_not(aux)]
        X = np.hstack([X_1, X_2])
        T = np.hstack([np.ones(X_1.shape[1]), np.ones(X_2.shape[1])*(-1)])
        aux2 = np.vstack([X, T]).T
        np.random.shuffle(aux2)
        X = aux2[:, :aux2.shape[1] - 1].T
        T = aux2[:, aux2.shape[1] - 1].T
        print('Comienza train de perceptron {0}'.format(i))
        per.train(X, T)
        print('Perceptron {0} entrenado\n'.format(i))
        perceptrones.append(per)
    return test_x, test_t, perceptrones

def test_mnist(test_x, test_t, perceptrones):
    mal = np.zeros(10)
    mal_esta = np.zeros(10)
    print('Comienza el test...')
    for i in range(test_x.shape[1]):
        arr = []
        for j in range(len(perceptrones)):
            per = perceptrones[j]
            val = per.eval_weights(test_x[:, i])
            if val > 0:
                mal_esta[j] = 1
            arr.append(val)
        np.argmax(np.array(arr))
        # print('{0:.2f}% completado'.format(i*100/test_x.shape[1]))
        if mal_esta[int(test_t[i])] == 1:
            mal_esta[int(test_t[i])] = 0
        else:
            mal_esta[int(test_t[i])] = 1
        mal += mal_esta
        mal_esta = np.zeros(10)
    for i in range(10):
        print('Perceptron {0} - Porcentaje de fallo en el test {1:.4f}%'.format(i, mal[i]*100/test_x.shape[1]))


if __name__ == '__main__':
    test_x, test_t, perceptrones = train_mnist()
    test_mnist(test_x, test_t.flatten(), perceptrones)