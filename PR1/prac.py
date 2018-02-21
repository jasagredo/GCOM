from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"


class LeastSquares(object):

    def __init__(self):
        self.w_tilde = None

    def train(self, X, t):
        x_vir = np.vstack([np.ones_like(X[0]), X])

        A = np.dot(x_vir, x_vir.T)
        b = np.dot(x_vir, t.T)

        self.w_tilde = np.linalg.solve(A, b)

        return self.w_tilde

    def classify(self, x):
        if self.w_tilde is None:
            print("No has entrenado el metodo")
        else:
            return (self.w_tilde[1:].T.dot(x) + self.w_tilde[0]).argmax(axis=0)


class LDA(object):

    def __init__(self):
        self.w = None

    def train(self, X, t):
        # Generacion del vector de medias
        mi_arr = []
        for i in range(0, t.shape[0]):
            mi_arr.append(np.mean(X[:, t[i] == 1], axis=1))
        mean = np.vstack(mi_arr)

        # Generacion de x_cent
        mi_arr = []
        for i in range(0, t.shape[0]):
            mi_arr.append(X[:, t[i] == 1].T - mean[i])
        x_cent = np.vstack(mi_arr)

        # Definicion de s_w
        s_w = x_cent.T.dot(x_cent)

        b = mean[0] - mean[1]

        self.w = np.linalg.solve(s_w, b)
        return self.w

    def classify(self, x):
        if self.w is None:
            print("No has entrenado el metodo")
        else:
            return (self.w[1:].T.dot(x) + self.w[0]).argmax(axis=0)

# W = np.array([[0.10123341,  1.2313048,  -0.33253821],
#               [-1.43416645, -0.33594547,  1.77011192],
#               [0.14541285, -0.15143434,  0.00602148]])

# xp = np.array([[-1.2, -1.3, -1,   -0.8, -0.85, -0.65,
#                 0.8, 0.9,  1,   1.1,  1.2,
#                 0.1, 0.15, -0.15, -0.35, -0.27,
#                 -0.25, -0.1, -0.1, 0.1, 0.2],
#                [0.82, 1,    0.78, 0.83, 1.2,   1,
#                 1.1, 0.95, 0.8, 0.92, 1.2,
#                 0.17, -0.2, 0.16, 0.19, -0.22,
#                 -1.25, -1.17, -0.85, -0.87, -1.2]])
#
# tp = np.array([[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]])
#
# # #Datos de entrenamiento


# plt.scatter(x[0], x[1])
# plt.show()

if __name__ == '__main__':
    # x = np.array([[0.1, 0.2, 0.1, 0.3, 0.25,
    #                0.1, 0.2, 0.1, 0.3, 0.25,
    #                0.6, 0.7, 0.65, 0.8],
    #               [8, 7, 7, 6, 9,
    #                1, 2, 1.5, 2, 1,
    #                6, 5, 8, 7]])
    # T = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

    x = np.array([[0.1, 0.15, 0.2, 0.3, 0.4, 0.35, 0.6, 0.65, 0.7, 0.8],
                  [0.8, 0.7, 0.5, 0.6, 0.3, 0.8, 0.7, 0.5, 0.6, 0.3]])
    t = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    ls = LDA()
    ls.train(x, t)
    print(ls.w)

    plt.scatter(x[0], x[1])
    plt.show()
