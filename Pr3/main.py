from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Clasificadores import *
from PCA import *
from Perceptron import *
from sklearn.datasets import load_breast_cancer
import sklearn.metrics as met
import time


def one_hot(x):
    can = np.zeros(nd)
    can[int(x)] = 1
    return np.array(can)


def obtenerMNIST():
    print("Cargando MNIST...")
    mnist = fetch_mldata('MNIST original', data_home = '~/Documents/Universidad/GCOM/Pr3')
    print("Mezclando datos...")
    mis_digitos = np.hstack([mnist.data, mnist.target.reshape(mnist.data.shape[0], 1)])
    np.random.shuffle(mis_digitos)
    ochenta = int(np.rint(mis_digitos.shape[0]*0.8))
    train = mis_digitos[:ochenta, :]
    train_x = train[:, :train.shape[1]-1].T
    train_t = train[:, train.shape[1]-1:]
    trainnf = train_t
    train_t = np.vstack(map(one_hot, train_t)).T
    test = mis_digitos[ochenta:, :]
    test_x = test[:, :test.shape[1]-1].T
    test_t = test[:, test.shape[1]-1:].flatten()
    print("Comprimiendo datos por PCA...")
    pca = PCA()
    data = pca.compresion(np.hstack([train_x, test_x]), tol=0.0001)
    train_x = data[:, :ochenta]
    test_x = data[:, ochenta:]
    print("\n")
    return test_t, test_x, train_t, train_x, trainnf


def obtenerBCWD():
    print("Cargando BCWD...")
    bcwd = load_breast_cancer()
    mis_digitos = np.hstack([bcwd.data, bcwd.target.reshape(bcwd.data.shape[0], 1)])
    print("Mezclando datos...")
    np.random.shuffle(mis_digitos)
    ochenta = int(np.rint(mis_digitos.shape[0] * 0.8))
    train = mis_digitos[:ochenta, :]
    train_x = train[:, :train.shape[1] - 1].T
    train_t = train[:, train.shape[1] - 1:]
    trainnf = train_t
    train_t = np.vstack(map(one_hot, train_t)).T
    test = mis_digitos[ochenta:, :]
    test_x = test[:, :test.shape[1] - 1].T
    test_t = test[:, test.shape[1] - 1:].flatten()
    print("Comprimiendo datos por PCA...")
    pca = PCA()
    data = pca.compresion(np.hstack([train_x, test_x]), tol=0.0001)
    train_x = data[:, :ochenta]
    test_x = data[:, ochenta:]
    return test_t, test_x, train_t, train_x, trainnf


def use_lda(test_t, test_x, train_t, train_x):
    print("Entrenando clasificador LDA...")
    lda = LDA_classifier(train_x, train_t)
    lda.train(0.0001)
    print("LDA preparado")
    print('Comienza el test...')
    res = lda.classify(test_x)
    print('#### Resultados:')
    print("Accuracy: {0}".format(met.accuracy_score(test_t, res)))
    print("Precision: {0}".format(met.precision_score(test_t, res, average='macro')))
    print("F1: {0}".format(met.f1_score(test_t, res, average='macro')))
    print("Recall: {0}".format(met.recall_score(test_t, res, average='macro')))


def use_ls(test_t, test_x, train_t, train_x):
    ls = LeastSquares()
    print("Entrenando Least Squares...")
    ls.train(train_x, train_t)
    print("Least Squares preparado")
    print("Comienza el test...")
    res = ls.classify(test_x)
    print('#### Resultados:')
    print("Accuracy: {0}".format(met.accuracy_score(test_t, res)))
    print("Precision: {0}".format(met.precision_score(test_t, res, average='macro')))
    print("F1: {0}".format(met.f1_score(test_t, res, average='macro')))
    print("Recall: {0}".format(met.recall_score(test_t, res, average='macro')))


def use_perceptron(test_t, test_x, train_t, train_x):
    per = Perceptron(test_x.shape[0], 1000)
    print("Entrenando un perceptron...")
    train_t = np.array(map(lambda x: 1 if x == 1 else -1, train_t))
    test_t = np.array(map(lambda x: 1 if x == 1 else -1, test_t))
    per.train(train_x, train_t)
    print("Perceptron preparado")
    print("Comienza el test...")
    res = []
    for i in range(test_x.shape[1]):
        res.append(per.eval(test_x[:, i]))
    print('#### Resultados:')
    print('#### Resultados:')
    print("Accuracy: {0}".format(met.accuracy_score(test_t, res)))
    print("Precision: {0}".format(met.precision_score(test_t, res, average='macro')))
    print("F1: {0}".format(met.f1_score(test_t, res, average='macro')))
    print("Recall: {0}".format(met.recall_score(test_t, res, average='macro')))


def use_10perceptron(test_t, test_x, train_t, train_x):
    print('Entrenando 10 perceptrones...')
    perceptrones = []
    for i in range(10):
        per = Perceptron(test_x.shape[0], 3)
        aux = np.equal(train_t, np.ones_like(train_t)*i).flatten()
        X_1 = train_x[:, aux]
        X_2 = train_x[:, np.logical_not(aux)]
        X = np.hstack([X_1, X_2])
        T = np.hstack([np.ones(X_1.shape[1]), np.ones(X_2.shape[1])*(-1)])
        aux2 = np.vstack([X, T]).T
        np.random.shuffle(aux2)
        X = aux2[:, :aux2.shape[1] - 1].T
        T = aux2[:, aux2.shape[1] - 1].T
        per.train(X, T)
        print('Perceptron {0} entrenado'.format(i))
        perceptrones.append(per)
    mal = np.zeros(10)
    mal_esta = np.zeros(10)
    print('Comienza el test...')
    res = []
    for i in range(test_x.shape[1]):

        arr = []
        for j in range(len(perceptrones)):
            per = perceptrones[j]
            val = per.eval_weights(test_x[:, i])
            if val > 0:
                mal_esta[j] = 1
            arr.append(val)
        res.append(np.argmax(np.array(arr)))
        if mal_esta[int(test_t[i])] == 1:
            mal_esta[int(test_t[i])] = 0
        else:
            mal_esta[int(test_t[i])] = 1
        mal += mal_esta
        mal_esta = np.zeros(10)
    for i in range(10):
        print('Perceptron {0} - Porcentaje de fallo en el test {1:.4f}%'.format(i, mal[i] * 100 / test_x.shape[1]))

    print('#### Resultados:')
    print("Accuracy: {0}".format(met.accuracy_score(test_t, res)))
    print("Precision: {0}".format(met.precision_score(test_t, res, average='macro')))
    print("F1: {0}".format(met.f1_score(test_t, res, average='macro')))
    print("Recall: {0}".format(met.recall_score(test_t, res, average='macro')))


if __name__ == '__main__':
    start = time.time()
    nd = 10
    test_t, test_x, train_t, train_x, trainnf = obtenerMNIST()
    use_lda(test_t, test_x, train_t, train_x)
    print('')
    use_ls(test_t, test_x, train_t, train_x)
    print('')
    use_10perceptron(test_t, test_x, trainnf, train_x)
    print('')

    nd = 2
    test_t, test_x, train_t, train_x, trainnf = obtenerBCWD()
    print('')
    use_lda(test_t, test_x, train_t, train_x)
    print('')
    use_ls(test_t, test_x, train_t, train_x)
    print('')
    use_perceptron(test_t, test_x, trainnf, train_x)
    end = time.time()
    print('Tiempo total empleado: {0:.4f}'.format(end-start))



