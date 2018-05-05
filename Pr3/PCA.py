from __future__ import print_function, division
import numpy as np
import time

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"


class PCA(object):

    def __init__(self):
        self.w = None

    def compresion(self, data, tol=0.1):
        '''data: D x N'''
        mean = np.mean(data, axis=1)
        assert tol > 0
        x_cent = data - mean[:, np.newaxis]
        print("Comienza SVD (esto puede tardar)...")
        start = time.time()
        u, s, _ = np.linalg.svd(x_cent, full_matrices=False)
        end = time.time()
        print("Ha tardado {0}".format(end - start))
        print("SVD terminado")
        # S = np.dot(np.dot(u, np.diag(s)),u.T)
        # autovectores son u[:,i], autovalores son s[i]. Ya estan ordenados
        s2 = map(lambda x: x**2, s)
        dp = s.shape[0]
        denom = np.sum(s2)
        val = 0
        for i in range(s.shape[0]-1, -1, -1):
            val += s2[i]
            if val/denom > tol:
                dp = i
                break
        self.w = u[:, 0:dp]
        print("Se ha solicitado una tolerancia de {0}".format(tol))
        print("PCA ha comprimido a {0} dimensiones".format(dp))
        return self.project(data) # D' x N

    def project(self, x):
        '''x: D x N'''
        return np.dot(self.w.T, x)