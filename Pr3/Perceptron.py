import numpy as np


class Perceptron:
    def __init__(self, D, iters):
        self.iters = iters
        self.size = D
        self.w_tilde = np.random.rand(self.size+1)

    def eval_weights(self, x):
        x_tilde = np.hstack([1, x])
        return self.w_tilde.dot(x_tilde)

    def eval(self, x):
        return np.sign(self.eval_weights(x))

    def train(self, X, T, w0=None, eta=0.1):
        ''' X: D x N
            T: N '''
        if w0 is not None:
            self.w_tilde = w0
        X_tilde = np.vstack([np.ones_like(X[0]), X])
        if self.iters != -1:
            for i in range(int(np.rint(self.iters))):
                for j in range(X_tilde.shape[1]):
                    elem = X_tilde[:, j]
                    elem_t = T[j]
                    estimate = self.eval(elem[1:])
                    if estimate * elem_t < 0:
                        self.w_tilde += eta * elem * elem_t
        else:
            w_tilde = self.w_tilde.copy()
            i = 0
            while True:
                for j in range(X_tilde.shape[1]):
                    elem = X_tilde[:, j]
                    elem_t = T[j]
                    estimate = self.eval(elem[1:])
                    if estimate * elem_t < 0:
                        self.w_tilde += eta * elem * elem_t
                i += 1
                if np.array_equal(self.w_tilde, w_tilde):
                    print("Numero de iters: {0}".format(i))
                    break
                else:
                    w_tilde = self.w_tilde.copy()

    def get_weights(self):
        return self.w_tilde
