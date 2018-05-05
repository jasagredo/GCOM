from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"


class LeastSquares(object):

    def __init__(self):
        self.w_tilde = None

    def train(self, X, t):
        """ X: D x N
            t: C x N
            None
         """

        x_tilde = np.vstack([np.ones_like(X[0]), X])

        A = np.dot(x_tilde, x_tilde.T).T
        b = np.dot(x_tilde, t.T)

        self.w_tilde = np.linalg.solve(A, b)

    def classify(self, x):
        """ x: D x N
            v[N]
        """
        if self.w_tilde is None:
            print("No has entrenado el metodo")
        else:
            if np.ndim(x) == 1:
                x_tilde = np.hstack([1, x])
            else:
                x_tilde = np.vstack([np.ones_like(x[0]),x])
            return (self.w_tilde.T.dot(x_tilde)).argmax(axis=0)


class Lda(object):
    def __init__(self):
        '''nc = C'''
        self.w = None
        self.c = None
        self.nc = 0
        self.x = None
        self.t = None
        self.mean = None
        self.sigma = None
        self.n = None
        self.nt = 0

    def train(self, X, t):
        ''' X: D x N
            t: C x N'''
        self.nc = t.shape[0]
        self.n = np.zeros(self.nc)
        if self.nc > 2:

            self.x = X
            self.t = t
            self.nt = X.shape[1]

            # Generacion del vector de medias
            mean = np.zeros((t.shape[0], X.shape[0]))
            for i in range(0, t.shape[0]):
                mean[i] = np.mean(X[:, t[i] == 1], axis=1)
            self.mean = mean
            total_mean = np.mean(X, axis=1)

            # Generacion de s_w
            s_w = np.zeros((X.shape[0], X.shape[0]))
            for i in range(0, t.shape[0]):
                elems = X[:, t[i] == 1]
                self.n[i] = elems.shape[1]
                x_i_m_i = elems - mean[i][:, np.newaxis]
                s_w += np.dot(x_i_m_i, x_i_m_i.T)

            # Generacion de s_b
            m_i_m = mean.T - total_mean[:, np.newaxis]
            s_b = np.dot(m_i_m, m_i_m.T)

            # Obtencion de los autovectores de (sw)^-1 sb
            s_w_s_b = np.linalg.inv(s_w).dot(s_b)
            v, w = np.linalg.eig(s_w_s_b)
            self.w = w[:, np.argmax(v)]

            # Preparacion de las sigmas
            self.sigma = np.zeros((t.shape[0], 1, 1))
            for i in range(0, t.shape[0]):
                elems = self.w.T.dot(X[:, t[i] == 1])
                x_i_m_i = elems - self.w.T.dot(mean[i])
                self.sigma[i] = np.dot(x_i_m_i, x_i_m_i.T)

        else:
            # Generacion del vector de medias
            mean = np.zeros((t.shape[0], X.shape[0]))
            for i in range(0, t.shape[0]):
                mean[i] = np.mean(X[:, t[i] == 1], axis=1)

            # Generacion de s_w
            s_w = np.zeros((X.shape[0], X.shape[0]))
            for i in range(0, t.shape[0]):
                elems = X[:, t[i] == 1]
                x_i_m_i = elems - mean[i][:, np.newaxis]
                s_w += np.dot(x_i_m_i, x_i_m_i.T)

            # Generacion de b
            b = mean[0] - mean[1]

            self.w = np.linalg.solve(s_w, b)
            self.w = self.w / np.linalg.norm(self.w)

            # Generacion del valor de corte
            mean = np.zeros(t.shape[0])
            prob = np.zeros(mean.shape)
            sig = np.zeros(mean.shape)
            for i in range(0, t.shape[0]):
                elems = X[:, t[i] == 1]
                mean[i] = np.mean(self.w.T.dot(elems))
                prob[i] = elems.shape[1] / X.shape[1]
                sig[i] = np.var(self.w.T.dot(elems))

            poly = np.zeros(3)
            poly[0] = 1 / (2 * (sig[1] ** 2)) - 1 / (2 * (sig[0] ** 2))
            poly[1] = mean[0] / (sig[0] ** 2) - mean[1] / (sig[1] ** 2)
            poly[2] = (mean[1] ** 2) / (2 * (sig[1] ** 2)) - (mean[0] ** 2) / (2 * (sig[0] ** 2)) + \
                  np.log(prob[0] / sig[0]) - np.log(prob[1] / sig[1])
            raices = np.roots(poly)
            if 2 * poly[0] * raices[0] + poly[1] > 0:
                self.c = raices[0]
            else:
                self.c = raices[1]

    def classify(self, x):
        '''x: D x N'''
        if self.w is None:
            print("No has entrenado el metodo")
        else:
            if self.nc > 2:
                res = []
                for pt in x.T:
                    valor = []
                    for k in range(0, self.nc):
                        x_cent = self.w.T.dot(pt - self.mean[k])
                        a1 = np.dot(x_cent, 1/self.sigma[k])
                        a2 = np.dot(a1, x_cent)
                        a3 = np.log(self.sigma[k])
                        a4 = 2 * np.log(self.n[k] / self.nt)
                        valor.append(a2 + a3 - a4)
                    res.append(np.argmin(valor))
                return res
            else:
                val = self.w.T.dot(x)
                res = map((lambda x: 1 if x < self.c else 0), val)
                return res
