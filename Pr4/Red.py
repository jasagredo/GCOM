from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


class multilayer_perceptron:
    @staticmethod
    def sigmoid(a):
        return 1/(1 + np.exp(-a))

    @staticmethod
    def sigmoid_d(a):
        return a*(1-a)

    @staticmethod
    def tanh(a):
        return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)

    @staticmethod
    def relu(a):
        if len(a) > 1:
            return np.array(map (lambda x: max(0, x), a))
        else:
            return max(0,a)

    @staticmethod
    def softmax(a):
        a -= np.max(a)
        denom = np.sum(map(np.exp, a))
        return map(lambda x: np.exp(x)/denom, a)

    @staticmethod
    def identity(a):
        return a

    def __init__(self, n_inputs, n_outputs, n_hidden, activation='sigmoid'):
        """ n_inputs: numero de entradas
            n_outputs: numero de salidas
            n_hidden: lista con las neuronas de cada capa oculta
        """
        self.res = [np.ones(n_inputs + 1)]
        for e in n_hidden:
            self.res.append(np.ones(e + 1))
        self.res.append(np.ones(n_outputs + 1))
        # res[i][j] va a tener el valor de la neurona j-esima de la capa i-esima
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_d = self.sigmoid_d
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'relu':
            self.activation = self.relu
        elif activation == 'softmax':
            self.activation = self.softmax
        elif activation == 'identity':
            self.activation = self.identity
        else:
            print('Error, funcion de activacion no conocida.')
            return
        # Pesos[i] representa la matriz de la pagina 42 de los apuntes [[1, 0..0],[b_i, w_i]]. Inicializamos a randoms
        self.pesos = [np.array([])]
        for i in range(1, len(self.res)):
            self.pesos.append(np.zeros((len(self.res[i]), len(self.res[i-1]))))
            self.pesos[i][0, 0] = 1
            self.pesos[i][1:, 0] = (np.random.rand(len(self.res[i]) - 1))/2 + 0.5
            self.pesos[i][1:, 1:] = (np.random.rand(len(self.res[i]) - 1, len(self.res[i-1]) - 1))/2 + 0.5


    def train(self, X, T, eta, epochs=1):
        """ X: D x N
            T: N
            eta: numero
            epochs: numero
        """
        a = zip(X.T, T)
        for _ in range(epochs):
            np.random.shuffle(a)
            for elem_x, elem_t in a:
                self.res[0] = np.hstack([1,elem_x]) # metemos el valor en las neuronas de entrada
                self.propagar()
                self.retropropagar(elem_t, eta)


    def propagar(self):
        for k in range(1, len(self.res)):
            self.res[k] = self.pesos[k].dot(self.res[k - 1])
            self.res[k] = np.hstack([1, self.activation(self.res[k][1:])])

    def retropropagar(self, T, eta):
        self.delta = []
        for k in reversed(range(1, len(self.res))):
            if k == len(self.res) - 1:
                self.delta.append(self.res[k][1:] - T)
            else:
                a = np.diag(self.res[k][1:])
                if len(self.delta[-1]) == 1:
                    b = self.pesos[k + 1][1:, 1:].T * self.delta[-1]
                else:
                    b = np.dot(self.pesos[k + 1][1:, 1:].T, self.delta[-1])
                self.delta.append(a.dot(b).flatten())

            self.pesos[k][1:, 0] = self.pesos[k][1:, 0] - eta * self.delta[-1]
            if len(self.delta[-1]) == 1:
                self.pesos[k][1:, 1:] = self.pesos[k][1:, 1:] - eta * self.delta[-1] * self.res[k - 1][1:]
            else:
                self.pesos[k][1:, 1:] = self.pesos[k][1:, 1:] - eta * np.outer(self.delta[-1], self.res[k - 1][1:])

    def classify(self, x):
        """ x: D """
        self.res[0] = np.hstack([1, x])
        self.propagar()
        return self.res[-1][1:]


a = multilayer_perceptron(2, 1, [2], activation='tanh')
X = np.array([[1,0,1,0], [1,1,0,0]])
T = np.array([0,1,1,0])
a.train(X, T, 0.005, epochs=75000)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
fondo = np.mgrid[-2:2:0.05, -2:2:0.05].reshape(2, 6400)
g = map(a.classify, fondo.T)
clase_fondo = map((lambda x: 'C' + str(x)), map((lambda x: 1 if x > 0.5 else 0), g))
ax.scatter(fondo[0], fondo[1], color=clase_fondo, alpha=0.2, s=5)
ax.scatter(X[0], X[1], color='k')
fig.canvas.draw()
plt.show()
