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

    def classify(self, x):
        if self.w_tilde is None:
            print("No has entrenado el metodo")
        else:
            if np.ndim(x) == 1:
                x_tilde = np.hstack([1, x])
            else:
                x_tilde = np.vstack([np.ones_like(x[0]),x])
            return (self.w_tilde.T.dot(x_tilde)).argmax(axis=0)

class LDA_classifier(object):
    def __init__(self, X, T):
        self.w = None
        self.c = None
        self.nc = 10
        self.x = X
        self.t = T
        self.mean = None
        self.n = np.zeros(10)
        self.nt = 0

    def train(self, epsilon):
        # Generacion del vector de medias
        self.nt = self.x.shape[1]
        mi_arr = []
        for i in range(0, self.t.shape[0]):
            media_i = np.mean(self.x[:, self.t[i] == 1], axis=1)
            mi_arr.append(media_i)
        mean = np.vstack(mi_arr)
        self.mean = mean
        total_mean = np.mean(self.x, axis=1)
        # Generacion de x_cent
        mi_S_w = []
        mi_S_b = []
        for i in range(0, self.t.shape[0]):
            elems = self.x[:, self.t[i] == 1]
            self.n[i] = elems.shape[1]
            x_i_m_i = elems.T - mean[i]
            mi_S_w.append(x_i_m_i)

            m_i_m = mean[i] - total_mean
            mi_S_b.append(elems.shape[1] * m_i_m.dot(m_i_m.T))

        x_cent = np.vstack(mi_S_w).T
        med_cent = np.vstack(mi_S_b).T
        s_w = x_cent.dot(x_cent.T)
        s_b = np.sum(med_cent)
        # TODO: peta porque s_w es una matriz singular (muchisimos ceros)
        s_w_s_b = np.linalg.inv(s_w).dot(s_b)
        u, s, _ = np.linalg.svd(s_w_s_b)
        # S = np.dot(np.dot(u, np.diag(s)),u.T)
        # autovectores son u[:,i], autovalores son s[i]. Ya estan ordenados
        s2 = map(lambda x: x ** 2, s)
        dp = s.shape[0]
        denom = np.sum(s2)
        val = 0
        for i in range(s.shape[0] - 1, -1, -1):
            val += s2[i]
            if val / denom > epsilon:
                dp = i
                break
        self.w = u[:, 0:dp]

    def classify(self, x):
        if self.w is None:
            print("No has entrenado el metodo")
        else:
            res = []
            for pt in x.T:
                valor = []
                for k in range(0, self.nc):
                    x_cent = self.w.T.dot(pt - self.mean[k])
                    elems = self.w.T.dot(self.x[:, self.t[k] == 1])
                    x_i_m_i = elems.T - self.w.T.dot(self.mean[k])
                    sigma = 1/self.n[k] * x_i_m_i.T.dot(x_i_m_i)
                    a1 = x_cent /sigma
                    a2 = a1 * x_cent
                    a3 = np.log(sigma)
                    a4 = 2*np.log(self.n[k]/self.nt)
                    valor.append(a2 + a3 - a4)
                res.append(np.argmin(valor))
            return res
