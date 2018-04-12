from __future__ import division, print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from Perceptron import *

mnist = fetch_mldata('MNIST original', data_home = '~/Documents/GCOM/Pr2')
data = mnist.data
target = mnist.target
mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
np.random.shuffle(mis_digitos)
ochenta = int(np.rint(mis_digitos.shape[0]*0.8))
train = mis_digitos[:ochenta, :]
train_x = train[:, :train.shape[1]-1]
train_t = train[:, train.shape[1]-1:]
test = mis_digitos[ochenta:, :]
test_x = test[:, :test.shape[1]-1]
test_t = test[:, test.shape[1]-1:]
perceptrones = []
for i in range(10):
    per = Perceptron(data.shape[1])
    aux = np.equal(train_t, np.ones_like(train_t)*i).reshape(train_x.shape[0])
    X_1 = train_x[aux, :]
    X_2 = train_x[np.logical_not(aux), :]
    X = np.vstack([X_1, X_2])
    T = np.hstack([np.ones(X_1.shape[0]), np.ones(X_2.shape[0])*(-1)]).T
    aux2 = np.hstack([X, T.reshape(T.shape[0],1)])
    np.random.shuffle(aux2)
    X = aux2[:, :aux2.shape[1] - 1]
    T = aux2[:, aux2.shape[1] - 1:]
    print('Comienza train de perceptron {0}'.format(i))
    per.train(X, T)
    perceptrones.append(per)
mal_c = 0
cl = 0
print('Comienza el test')
for i in range(test_x.shape[0]):
    arr = []
    for per in perceptrones:
        arr.append(per.eval(test_x[i]))
    sal = np.argmax(np.array(arr))
    cl = cl + 1
    print('{0}% completado'.format(i*100/test_x.shape[0]))
    if test_t[i] != sal:
        mal_c = mal_c + 1
print('Porcentaje de fallo de clasificacion en el test {0}'.format(mal_c*100/cl))