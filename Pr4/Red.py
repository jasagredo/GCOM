from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(a):
    if type(a) is not np.ndarray:
        return np.array(1/(1 + np.exp(-a)) if a > 0 else np.exp(a)/(np.exp(a)+1))
    else:
        return np.array(map( lambda a: 1/(1 + np.exp(-a)) if a > 0 else np.exp(a)/(np.exp(a)+1),a))


def sigmoid_d(a):
    return np.array(map(lambda a: sigmoid(a) * (1 - sigmoid(a)), a))


def tanh(a):
    return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)


def tanh_d(a):
    return 1- (tanh(a))**2


def relu(a):
    if len(a) > 1:
        return np.array(map (lambda x: np.maximum(0, x), a))
    else:
        return np.maximum(0,a)


def relu_d(a):
    if len(a) > 1:
        return np.array(map (lambda i :np.ones(1) if i > 0 else np.zeros(1), a))
    else:
        return np.ones(1) if a > 0 else np.zeros(1)


def softmax(a):
    a -= np.max(a)
    denom = np.sum(map(np.exp, a))
    return map(lambda x: np.exp(x)/denom, a)


def identity(a):
    return a


class multilayer_perceptron:

    def __init__(self, n_inputs, n_outputs, n_hidden, activation='sigmoid'):
        """ n_inputs: numero de entradas
            n_outputs: numero de salidas
            n_hidden: lista con las neuronas de cada capa oculta
        """
        self.a = [np.ones(n_inputs)]
        self.a_d = [np.ones(n_inputs)]
        self.z = [np.ones(n_inputs)]
        for e in n_hidden:
            self.a.append(np.ones(e))
            self.a_d.append(np.ones(e))
            self.z.append(np.ones(e))
        self.a.append(np.ones(n_outputs))
        self.a_d.append(np.ones(n_outputs))
        self.z.append(np.ones(n_outputs))
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_d = sigmoid_d
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_d = tanh_d
        elif activation == 'relu':
            self.activation = relu
            self.activation_d = relu_d
        elif activation == 'softmax':
            self.activation = self.softmax
        elif activation == 'identity':
            self.activation = self.identity
        else:
            print('Error, funcion de activacion no conocida.')
            return
        self.pesos = [np.array([])]
        self.sesgos = [np.array([])]
        for i in range(1, len(self.a)):
            self.sesgos.append(np.random.rand(len(self.a[i])))
            self.pesos.append(np.random.rand(len(self.a[i]), len(self.a[i - 1])))

    def train(self, X, T, eta, epochs=1):
        """ X: D x N
            T: N
            eta: numero
            epochs: numero
        """
        a = zip(X.T, T)
        for i in range(epochs):
            print('Epoch {0}'.format(i))
            np.random.shuffle(a)
            for elem_x, elem_t in a:
                self.z[0] = elem_x
                self.propagar()
                self.retropropagar(elem_t, eta)

    def propagar(self):
        for k in range(1, len(self.a)):
            self.a[k]   = self.pesos[k].dot(self.z[k - 1]) + self.sesgos[k]  # ak
            self.a_d[k] = self.activation_d(self.a[k])  # h'(ak)
            self.z[k]   = self.activation(self.a[k])  # h(ak)

    def retropropagar(self, T, eta):
        for k in reversed(range(1, len(self.a))):
            if k == len(self.a) - 1:
                delta = self.z[k] - T
            else:
                a = np.diag(self.a_d[k])
                if delta.shape[0] == 1:
                    b = self.pesos[k + 1].T * delta
                else:
                    b = np.dot(self.pesos[k + 1].T, delta)
                delta = a.dot(b).flatten()

            self.sesgos[k] = self.sesgos[k] - eta * delta
            if delta.shape[0] == 1:
                self.pesos[k] = self.pesos[k] - eta * delta * self.z[k - 1]
            else:
                self.pesos[k] = self.pesos[k] - eta * np.outer(delta, self.z[k - 1])

    def classify(self, x):
        """ x: D """
        self.z[0] = x
        self.propagar()
        return self.z[-1]


