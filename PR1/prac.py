from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"


class LeastSquares(object):

    def __init__(self):
        self.w_tilde = None

    def train(self, X, t):
        x_tilde = np.vstack([np.ones_like(X[0]), X])

        A = np.dot(x_tilde, x_tilde.T)
        b = np.dot(x_tilde, t.T)

        self.w_tilde = np.linalg.solve(A, b)

        return self.w_tilde

    def classify(self, x):
        if self.w_tilde is None:
            print("No has entrenado el metodo")
        else:
            if np.ndim(x) == 1:
                x_tilde = np.hstack([1, x])
            else:
                x_tilde = np.vstack([np.ones_like(x[0]),x])
            return (self.w_tilde.T.dot(x_tilde)).argmax(axis=0)


class LDA(object):

    def __init__(self):
        self.w = None
        self.c = None

    def train(self, X, t):
        # Generacion del vector de medias
        mi_arr = []
        for i in range(0, t.shape[0]):
            media_i = np.mean(X[:, t[i] == 1], axis=1)
            mi_arr.append(media_i)
        mean = np.vstack(mi_arr)
        # Generacion de x_cent
        mi_arr = []
        for i in range(0, t.shape[0]):
            x_i_m_i = X[:, t[i] == 1].T - mean[i]
            mi_arr.append(x_i_m_i)
        x_cent = np.vstack(mi_arr).T

        # Definicion de s_w
        s_w = x_cent.dot(x_cent.T)

        b = mean[0] - mean[1]

        self.w = np.linalg.solve(s_w, b)
        self.w = self.w / np.linalg.norm(self.w)
        return self.w

    def get_root(self, X, t):
        medias = []
        probs = []
        sigs = []
        for i in range(0, t.shape[0]):
            elems = X[:, t[i] == 1]
            medias.append(np.mean(self.w.T.dot(elems)))

            probs_i = elems.shape[1]/X.shape[1]
            probs.append(probs_i)

            sigs.append(np.var(self.w.T.dot(elems)))

        mean = np.vstack(medias)
        prob = np.vstack(probs)
        sig = np.vstack(sigs)

        a_2 = 1/(2*(sig[1]**2)) - 1/(2*(sig[0]**2))
        a_1 = mean[0]/(sig[0]**2) - mean[1]/(sig[1]**2)
        a_0 = (mean[1]**2)/(2*(sig[1]**2)) - (mean[0]**2)/(2*(sig[0]**2)) + np.log(prob[0]/sig[0]) - np.log(prob[1]/sig[1])
        poly = np.hstack([a_2, a_1, a_0])
        raices = np.roots(poly)
        if 2*a_2*raices[0] + a_1 > 0:
            self.c = raices[0]
        else:
            self.c = raices[1]

    def classify(self, x):
        if self.w is None:
            print("No has entrenado el metodo")
        else:
            val = self.w.T.dot(x)
            res = map((lambda x: 1 if x < self.c else 0), val)
            return res
