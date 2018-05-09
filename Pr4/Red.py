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
        return max(0, a)        #TODO peta porque si a es vector, max es ambiguo

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
        self.res = [np.ones(n_inputs)]
        for e in n_hidden:
            self.res.append(np.ones(e))
        self.res.append(np.ones(n_outputs))
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
            self.pesos.append(np.zeros((len(self.res[i]) + 1, len(self.res[i-1]) + 1)))
            self.pesos[i][0, 0] = 1
            self.pesos[i][1:, 0] = np.random.rand(len(self.res[i]))
            self.pesos[i][1:, 1:] = np.random.rand(len(self.res[i]), len(self.res[i-1]))


    def train(self, X, T, eta, epochs=1):
        """ X: D x N
            T: N
            eta: numero
            epochs: numero
        """
        for _ in range(epochs):
            print('')
            for i in range(X.shape[1]):
                self.res[0] = np.hstack([1,X[:, i]]) # metemos el valor en las neuronas de entrada
                self.propagar()
                if T.ndim == 1:
                    self.retropropagar(T[i], eta)
                else:
                    self.retropropagar(T[:, i], eta)
                print(self.pesos)
                print('')
                print(self.res)

    def propagar(self):
        for k in range(1, len(self.res)):
            self.res[k] = self.pesos[k].dot(self.res[k - 1]) # calculamos salida con los pesos
            self.res[k] = np.hstack([1, self.activation(self.res[k][1:])]) # procesamos el dato por la func de activacion y le anadimos un 1 encima

    def retropropagar(self, T, eta):
        self.delta = []
        for k in reversed(range(len(self.res))):
            if k == len(self.res) - 1:
                self.delta.append(self.res[k][1:] - T) # en el caso de la ultima capa, restamos y - t
                print(sum(self.delta[0]))
            else:
                a = np.diag(self.res[k][1:]) # a = diag(z^k)
                b = np.dot(self.pesos[k + 1][1:, 1:].T, self.delta[-1]) # b = w^(k+1).T * d^(k+1)
                self.delta.append(a.dot(b))
                # Notese que en estas dos operaciones las k escritas difieren en uno de las k de los comentarios
                self.pesos[k + 1][1:, 0] = self.pesos[k + 1][1:, 0] - eta * self.delta[-2] # bias_k = pesos_k[1:, 0] lo actualizamos con eta menos su gradiente que es d^(k)
                if len(self.delta[-2]) == 1:
                    self.pesos[k + 1][1:, 1:] = self.pesos[k+1][1:, 1:]- eta * self.delta[-2] * self.res[k][1:] # w_k = pesos_k[1:, 1:] lo actualizamos con eta menos su gradiente que es d^(k) * z_(k-1)
                else:
                    self.pesos[k + 1][1:, 1:] = self.pesos[k+1][1:, 1:] - eta * np.outer(self.delta[-2], self.res[k][1:])

    def classify(self, x):
        """ x: D """
        self.res[0] = np.hstack([1, x])
        self.propagar()
        return self.res[-1][1:]

class aaa:
    def __init__(self):
        a = multilayer_perceptron(2, 2, [2], activation='sigmoid')
        X = np.array([[1,0,1,0], [1,1,0,0]])
        T = np.array([0,1,1,0])
        a.train(X, T, 0.8, epochs=5000)
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        fondo = np.mgrid[-2:2:0.05, -2:2:0.05].reshape(2, 6400)
        clase_fondo = map((lambda x: 'C' + str(x)), map(np.argmax, map(a.classify, fondo.T)))
        ax.scatter(fondo[0], fondo[1], color=clase_fondo, alpha=0.2, s=5)
        fig.canvas.draw()
        plt.show()

a = aaa()