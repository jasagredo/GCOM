from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Clasificadores import *
from PCA import *
from Perceptron import *
from sklearn.datasets import load_breast_cancer
import sklearn.metrics as met

nd = 10

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
    return test_t, test_x, train_t, train_x


def use_lda(test_t, test_x, train_t, train_x):
    print("Entrenando clasificador LDA...")
    lda = LDA_classifier(train_x, train_t)
    lda.train(0.0001)
    print("LDA preparado")
    cl = test_x.shape[1]
    print('Comienza el test...')
    res = lda.classify(test_x)
    print("Accuracy: {0}".format(met.accuracy_score(test_t, res)))
    print("Precision: {0}".format(met.precision_score(test_t, res, average='macro')))
    print("F1: {0}".format(met.f1_score(test_t, res, average='macro')))
    print("Recall: {0}".format(met.recall_score(test_t, res, average='macro')))
    mal = len(filter((lambda x: x[0] != x[1]), zip(res, test_t)))
    print('#### Resultados:')
    print('Mal clasificado: {0:.4f}\n'.format(mal * 100 / cl))


def use_ls(test_t, test_x, train_t, train_x):
    ls = LeastSquares()
    print("Entrenando Least Squares...")
    ls.train(train_x, train_t)
    print("Least Squares preparado")
    cl = test_x.shape[1]
    print("Comienza el test...")
    res = ls.classify(test_x)
    mal = len(filter((lambda x: x[0] != x[1]), zip(res, test_t)))
    print('#### Resultados:')
    print('Mal clasificado: {0:.4f}\n'.format(mal * 100 / cl))


def use_perceptron(test_t, test_x, train_t, train_x):
    ls = Perceptron(test_x.shape[0], 1000)
    print("Entrenando Perceptron...")
    train_t = np.array(map(lambda x: 1 if x[1] == 1 else -1, train_t.T))
    ls.train(train_x, train_t)
    print("Perceptron preparado")
    cl = test_x.shape[1]
    print("Comienza el test...")
    res = ls.eval_weights(test_x)
    mal = len(filter((lambda x: np.sign(x[0]) != np.sign(x[1])), zip(res, test_t)))
    print('#### Resultados:')
    print('Mal clasificado: {0:.4f}\n'.format(mal * 100 / cl))

def use_10perceptron(test_t, test_x, train_t, train_x):
    perceptrones = []
    for i in range(10):
        per = Perceptron(test_x.shape[0], 50)
        aux = np.equal(train_t, np.ones_like(train_t) * i).reshape(train_x.shape[0])
        X_1 = train_x[aux, :]
        X_2 = train_x[np.logical_not(aux), :]
        X = np.vstack([X_1, X_2])
        T = np.hstack([np.ones(X_1.shape[0]), np.ones(X_2.shape[0]) * (-1)]).T
        aux2 = np.hstack([X, T.reshape(T.shape[0], 1)])
        np.random.shuffle(aux2)
        X = aux2[:, :aux2.shape[1] - 1]
        T = aux2[:, aux2.shape[1] - 1:]
        print('Comienza train de perceptron {0}'.format(i))
        per.train(X.T, T)
        perceptrones.append(per)
        # TODO: falta por crear el eval de todos estos perceptrones y comprobarlo para sacar errores


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
    train_t = np.vstack(map(one_hot, train_t)).T
    test = mis_digitos[ochenta:, :]
    test_x = test[:, :test.shape[1] - 1].T
    test_t = test[:, test.shape[1] - 1:].flatten()
    print("Comprimiendo datos por PCA...")
    pca = PCA()
    data = pca.compresion(np.hstack([train_x, test_x]), tol=0.0001)
    train_x = data[:, :ochenta]
    test_x = data[:, ochenta:]
    print("\n")
    return test_t, test_x, train_t, train_x


if __name__ == '__main__':
    nd = 10
    test_t, test_x, train_t, train_x = obtenerMNIST()
    use_lda(test_t, test_x, train_t, train_x)
    use_ls(test_t, test_x, train_t, train_x)
    #use_10perceptron(test_t, test_x, train_t, train_x)

    #nd = 2
    #test_t, test_x, train_t, train_x = obtenerBCWD()
    #use_lda(test_t, test_x, train_t, train_x)
    #use_ls(test_t, test_x, train_t, train_x)
    #use_perceptron(test_t, test_x, train_t, train_x)



