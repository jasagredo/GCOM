from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mnist import *
from matplotlib.widgets import Button

class DibujaNumeros:

    def __init__(self, fig, ax, bx, bx2):
        self.mat = np.zeros((28,28))
        self.fig = fig
        self.ax = ax
        self.bx = bx
        self.bx2 = bx2
        self.ax.imshow(self.mat)
        self.fig.canvas.draw()
        self.x0 = None
        self.y0 = None
        self.press_event = None
        _ , self.perceptrones = train_mnist()

        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

    def on_press(self, event):
        if self.ax == event.inaxes:
            self.x0, self.y0 = int(np.rint(event.xdata)), int(np.rint(event.ydata))
            self.press_event = event
            if self.mat[self.y0, self.x0] == 0:
                self.mat[self.y0, self.x0] = 1
            self.ax.imshow(self.mat)
            self.fig.canvas.draw()
        elif self.bx == event.inaxes:
            self.test()
        elif self.bx2 == event.inaxes:
            self.mat = np.zeros((28,28))
            self.ax.imshow(self.mat)
            self.fig.canvas.draw()

    def on_move(self, event):
        if self.press_event is not None:
            x0, y0 = int(np.rint(event.xdata)), int(np.rint(event.ydata))
            if (self.x0 != x0 or self.y0 != y0) and self.mat[y0, x0] == 0:
                self.mat[y0, x0] = 1

    def on_release(self, event):
        if self.press_event is not None:
            self.ax.imshow(self.mat)
            self.fig.canvas.draw()
        self.press_event = None

    def test(self):
        arr = []
        for j in range(len(self.perceptrones)):
            per = self.perceptrones[j]
            val = per.eval_weights(self.mat.reshape(784))
            arr.append(val)
        print("Los perceptrones dicen que es un {0}".format(np.argmax(np.array(arr))))

if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    bax = plt.axes([0.1, 0.15, 0.8, 0.04])
    boton = Button(bax, 'Clasifica')
    bax2 = plt.axes([0.1, 0.05, 0.8, 0.04])
    boton2 = Button(bax2, 'Limpia')
    start = DibujaNumeros(fig, ax, bax, bax2)
    plt.show()