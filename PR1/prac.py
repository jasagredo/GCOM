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

class LDA_Multiclass(object):
    def __init__(self, nc):
        self.w = None
        self.c = None
        self.nc = nc
        self.x = None
        self.t = None
        self.mean = None
        self.n = np.zeros(nc)
        self.nt = 0

    def train(self, X, t):
        if self.nc > 2:
            # Generacion del vector de medias
            self.x = X
            self.t = t
            self.nt = X.shape[1]
            mi_arr = []
            for i in range(0, t.shape[0]):
                media_i = np.mean(X[:, t[i] == 1], axis=1)
                mi_arr.append(media_i)
            mean = np.vstack(mi_arr)
            self.mean = mean
            total_mean = np.mean(X, axis=1)
            # Generacion de x_cent
            mi_S_w = []
            mi_S_b = []
            for i in range(0, t.shape[0]):
                elems = X[:, t[i] == 1]
                self.n[i] = elems.shape[1]
                x_i_m_i = elems.T - mean[i]
                mi_S_w.append(x_i_m_i)

                m_i_m = mean[i] - total_mean
                mi_S_b.append(elems.shape[1] * m_i_m.dot(m_i_m.T))

            x_cent = np.vstack(mi_S_w).T
            med_cent = np.vstack(mi_S_b).T
            s_w = x_cent.dot(x_cent.T)
            s_b = np.sum(med_cent)
            s_w_s_b = np.linalg.inv(s_w).dot(s_b)
            v, w = np.linalg.eig(s_w_s_b)
            self.w = w[:, np.argmax(v)]

        else:
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


            medias = []
            probs = []
            sigs = []
            for i in range(0, t.shape[0]):
                elems = X[:, t[i] == 1]
                medias.append(np.mean(self.w.T.dot(elems)))

                probs_i = elems.shape[1] / X.shape[1]
                probs.append(probs_i)

                sigs.append(np.var(self.w.T.dot(elems)))

            mean = np.vstack(medias)
            prob = np.vstack(probs)
            sig = np.vstack(sigs)

            a_2 = 1 / (2 * (sig[1] ** 2)) - 1 / (2 * (sig[0] ** 2))
            a_1 = mean[0] / (sig[0] ** 2) - mean[1] / (sig[1] ** 2)
            a_0 = (mean[1] ** 2) / (2 * (sig[1] ** 2)) - (mean[0] ** 2) / (2 * (sig[0] ** 2)) + np.log(
                prob[0] / sig[0]) - np.log(prob[1] / sig[1])
            poly = np.hstack([a_2, a_1, a_0])
            print(poly)
            raices = np.roots(poly)
            print(raices)
            if 2 * a_2 * raices[0] + a_1 > 0:
                self.c = raices[0]
            else:
                self.c = raices[1]

    def classify(self, x):
        if self.w is None:
            print("No has entrenado el metodo")
        else:
            if self.nc > 2:
                res = []
                for pt in x.T:
                    valor = []
                    for k in range(0, self.nc):
                        x_cent = pt - self.mean[k]
                        elems = self.x[:, self.t[k] == 1]
                        x_i_m_i = elems.T - self.mean[k]
                        sigma = 1/self.n[k] * x_i_m_i.T.dot(x_i_m_i)
                        a1 = x_cent.reshape(2,1).T.dot(np.linalg.inv(sigma))
                        a2 = a1.dot(x_cent)
                        a3 = np.log(np.linalg.det(sigma))
                        a4 = 2*np.log(self.n[k]/self.nt)
                        valor.append(a2 + a3 - a4)
                    res.append(np.argmin(valor))
                return res
            else:
                val = self.w.T.dot(x)
                res = map((lambda x: 1 if x < self.c else 0), val)
                return res
