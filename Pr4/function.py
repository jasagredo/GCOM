from Red import *

a = multilayer_perceptron(1, 1, [3], activation='sigmoid',  coste='regresion')
X = np.mgrid[0:10:1]
T = np.array(map(lambda x: np.sin(x), X))
a.train(X, T, 0.01, epochs=10000)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(0, 20)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
fondo = np.mgrid[0:20:0.05]
res = np.array(map(a.classify, fondo.T))
ax.scatter(X, T, color='k')
ax.plot(fondo, res.reshape(400), color='b')
fig.canvas.draw()
plt.show()