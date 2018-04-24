from __future__ import division, print_function
import numpy as np
from Perceptron import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


datos_iniciales = np.array([[0, 1, 0, 1],[0, 0, 1, 1]]).T
t = np.array([-1, 1, 1, -1])

def transformacion(elem):
    return np.array([elem[0]**2, np.sqrt(2)*elem[0]*elem[1], elem[1]**2])

a = np.array(map(transformacion, datos_iniciales))

per = Perceptron(3, 8)
per.train(a, t, w0=np.array([0.16943406, 0.64104446, 0.01417507, 0.95339745]))
#per.train(a, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = a.T[0][t == 1]
y1 = a.T[1][t == 1]
z1 = a.T[2][t == 1]
x2 = a.T[0][t == -1]
y2 = a.T[1][t == -1]
z2 = a.T[2][t == -1]


ax.scatter(x1, y1, z1, c='r', marker='o')
ax.scatter(x2, y2, z2, c='b', marker='^')

fondo = np.mgrid[0:1:0.05, 0:1.5:0.05, 0:1:0.05].reshape(3, 12000)

clase_fondo = map((lambda x: 'C1' if per.eval_weights(x) > 0 else 'C0'), fondo.T)
ax.scatter(fondo[0], fondo[1], fondo[2], color=clase_fondo, alpha=0.2, s=5)

plt.show()
