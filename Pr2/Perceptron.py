import numpy as np


class Perceptron:
    def __init__(self, D):
        self.size = D
        self.w_tilde = np.zeros(D+1)

    def eval_weights(self, x):
        x_tilde = np.hstack([1, x])
        return self.w_tilde.dot(x_tilde)

    def eval(self, x):
        return self.w_tilde.dot(np.hstack([1,x]))

    def train(self, X, T, w0 = None, eta = 0.1):
        if w0 == None:
            w0 = np.zeros(self.size+1)
        X_tilde = np.vstack([np.ones_like(X.T[0]), X.T])
        wxt = np.multiply(w0.dot(X_tilde), T)
        zeros = np.zeros_like(wxt)
        clasif = np.greater(wxt, 0)
        while np.any(np.logical_not(clasif)):
            elem = X_tilde[:, np.logical_not(clasif)]
            print '{} elementos mal clasificados'.format(elem.shape[1])
            sus_t = T[np.logical_not(clasif)]
            escogido = np.random.randint(elem.shape[1])
            w0 = w0 + eta*elem[:, escogido]*sus_t[escogido]
            wxt = np.multiply(w0.dot(X_tilde), T)
            zeros = np.zeros_like(wxt)
            clasif = np.greater(wxt, 0)
        self.w_tilde = w0

    def get_weights(self):
        return self.w_tilde[1:]




