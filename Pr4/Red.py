import numpy as np


class multilayer_perceptron:
    @staticmethod
    def sigmoid(a):
        return 1/(1 + np.exp(-a))

    @staticmethod
    def tanh(a):
        return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)

    @staticmethod
    def relu(a):
        return max(0,a)

    @staticmethod
    def softmax(a):
        h = np.max(a)
        a -= np.max(a)
        denom = np.sum(map(np.exp, a))
        return map(lambda x: np.exp(x)/denom, a)

    def __init__(self, n_inputs, n_outputs, n_hidden, activation='sigmoid'):
        self.res = [np.ones(n_inputs)]
        self.res.append(np.ones(n_hidden))
        self.res.append(np.ones(n_outputs))
        if activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'relu':
            self.activation = self.relu
        elif activation == 'softmax':
            self.activation = self.softmax
        else:
            print('Error, funcion de activacion no conocida.')
            return
        self.pesos = ['dummy']
        for i in range(1, len(self.res)):
            self.pesos.append(np.zeros((len(self.res[i]) + 1, len(self.res[i-1]) + 1)))
            self.pesos[i][0, 0] = 1
            self.pesos[i][1:, 0] = np.random.rand(len(self.res[i]))
            self.pesos[i][1:, 1:] = np.random.rand(len(self.res[i]), len(self.res[i-1]))

    def train(self, X, T, eta, epochs=1):
        pass

    def classify(self,x):
        pass

