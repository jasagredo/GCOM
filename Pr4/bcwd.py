from sklearn.datasets import load_breast_cancer
import sklearn.metrics as met
import numpy as np
from Red import *

def one_hot(x):
    can = np.zeros(2)
    can[int(x)] = 1
    return np.array(can)

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
    return test_t, test_x, train_t, train_x, trainnf

test_t, test_x, train_t, train_x, trainnf = obtenerBCWD()
a = multilayer_perceptron(30, 2, [8, 8], activation='relu', coste='binaria')
a.train(train_x, train_t.T, 0.1, epochs=100)

res = []
for elem in test_x.T:
    m = a.classify(elem)
    res.append(np.argmax(m))

print(met.precision_score(test_t, res, average='micro'))
print(met.f1_score(test_t, res, average='micro'))
print(met.recall_score(test_t, res, average='micro'))
print(met.accuracy_score(test_t, res))