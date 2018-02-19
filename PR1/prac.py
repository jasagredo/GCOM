from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"

class Practica1:

    def __init__(self): pass

    def classify(self, x, w):
        return np.add(np.dot(np.transpose(w[1:]), x), w[0]).argmax(axis=0) + 1

    def least_squares(self, X, t):
        x_vir = np.vstack([np.ones_like(X[0]), X])
        return np.dot(np.dot(np.linalg.inv(np.dot(x_vir, x_vir.T)), x_vir), t.T)

    def lda(self, X, t):
        m3an = np.vstack([np.mean(X[:, t[0] == 1], axis=1),
                          np.mean(X[:, t[1] == 1], axis=1)])
        x_cent = np.vstack([X[:, t[0] == 1].T - m3an[0], X[:, t[1] == 1].T - m3an[1]])
        s_w = np.trace(np.split(np.array(np.split(np.outer(x_cent, x_cent), X.shape[1], axis=0)), X.shape[1], axis=2))
        return np.dot(np.linalg.inv(s_w), np.subtract(m3an[0], m3an[1]))



# W = np.array([[0.10123341,  1.2313048,  -0.33253821],
#               [-1.43416645, -0.33594547,  1.77011192],
#               [0.14541285, -0.15143434,  0.00602148]])


# x = np.array([[0.3, 0.6, 0.5, 0.4, 0.45,
#                0.1, 0.2, 0.1, 0.3, 0.25,
#                0.6, 0.7, 0.65, 0.8],
#               [8, 7, 7, 6, 9,
#                1, 2, 1.5, 2, 1,
#                13, 12, 14, 15]])
# T = np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,1,1,1,1,1,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0,1,1,1,1]])

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
# x = np.array([[0.1, 0.15, 0.2, 0.3, 0.4, 0.35, 0.6, 0.65, 0.7, 0.8],
#                [0.8, 0.7,  0.5, 0.6, 0.3, 0.8,  0.7, 0.5, 0.6, 0.3]])
# #
#
# # #Clases de los datos
# t = np.array([[1,1,1,1,1,0,0,0,0,0],
#               [0,0,0,0,0,1,1,1,1,1]])

