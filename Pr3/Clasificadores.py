from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np
from PCA import *

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"


class LeastSquares(object):

    def __init__(self):
        self.w_tilde = None

    def train(self, X, t):
        ''' X: D x N
            t: C x N'''
        x_tilde = np.vstack([np.ones_like(X[0]), X])

        A = np.dot(x_tilde, x_tilde.T)
        b = np.dot(x_tilde, t.T)

        self.w_tilde = np.linalg.solve(A, b)

    def classify(self, x):
        ''' x: D x N'''
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
        ''' X: D x N
            t: C x N'''
        self.w = None
        self.c = None
        self.nc = T.shape[0]
        self.x = X
        self.t = T
        self.mean = None
        self.n = np.zeros(T.shape[0])
        self.nt = X.shape[1]
        self.sigma = None

    def train(self, epsilon):

        # Generacion del vector de medias
        mean = np.zeros((self.t.shape[0], self.x.shape[0]))
        for i in range(0, self.t.shape[0]):
            mean[i] = np.mean(self.x[:, self.t[i] == 1], axis=1)
        self.mean = mean
        total_mean = np.mean(self.x, axis=1)
        # Generacion de x_cent
        s_w = np.zeros((self.x.shape[0], self.x.shape[0]))
        s_b = np.zeros(s_w.shape)
        for i in range(0, self.t.shape[0]):
            elems = self.x[:, self.t[i] == 1].T
            self.n[i] = elems.shape[0]
            for elem in elems:
                x_i_m_i = (elem - mean[i]).T
                s_w += np.outer(x_i_m_i, x_i_m_i)

            m_i_m = mean[i] - total_mean
            s_b += (elems.shape[1] * np.outer(m_i_m, m_i_m))

        s_w_s_b = np.linalg.inv(s_w).dot(s_b)
        print("Comienza SVD (esto puede tardar)...")
        u, s, _ = np.linalg.svd(s_w_s_b.T, full_matrices=False)
        print("SVD terminado!")
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
        if dp == 0:
            dp = 1
        self.w = u[:, 0:dp]
        print("LDA ha reducido a {0} dimensiones".format(dp))
        self.sigma = np.zeros((self.nc, self.w.shape[1], self.w.shape[1]))
        for k in range(0, self.nc):
            elems = self.w.T.dot(self.x[:, self.t[k] == 1]).T
            for elem in elems:
                x_i_m_i = elem - self.w.T.dot(self.mean[k])
                self.sigma[k] += (1 / self.n[k]) * np.outer(x_i_m_i, x_i_m_i)

    def classify(self, x):
        '''x: D x N'''
        if self.w is None:
            print("No has entrenado el metodo")
        else:
            res = []
            for pt in x.T:
                valor = []
                for k in range(0, self.nc):
                    x_cent = self.w.T.dot(pt - self.mean[k])
                    a1 = np.dot(x_cent,np.linalg.inv(self.sigma[k]))
                    a2 = np.dot(a1, x_cent)
                    a3 = np.log(np.linalg.det(self.sigma[k]))
                    a4 = 2*np.log(self.n[k]/self.nt)
                    valor.append(a2 + a3 - a4)
                res.append(np.argmin(valor))
            return res