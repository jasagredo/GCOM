import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from prac import *


class CreatePoints(object):
    """Draw and drag points.

    Use left button to place points.
    Points are draggable. Use right button
    to disconnect and print and return
    the coordinates of the points.

    Args:
         fig: matplotlib figure
         ax: matplotlib axes
    """
    
    def __init__(self, fig, ax, axb1, axb2, axb3, axb4):
        self.circle_list = []
        self.x = []
        self.t = np.array([])
        self.conteo_clase_max = 0
        self.clase_max = 0
        self.clase_actual = 0

        self.metodo = None
        self.x0 = None
        self.y0 = None

        self.fig = fig
        self.ax = ax

        self.fondo = np.mgrid[-20:20,-20:20].reshape(2, 1600)
        
        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        self.press_event = None
        self.current_circle = None
        self.current_x = None

        self.axb1 = axb1
        self.axb2 = axb2
        self.axb3 = axb3
        self.axb4 = axb4
        Button(self.axb1, 'Least Squares')
        Button(self.axb2, 'LDA')
        self.bclasess = Button(self.axb3, 'Clase +')
        self.bclasesb = Button(self.axb4, 'Clase -')
        self.ax.set_title('Introduciendo puntos para la clase 0')

    def on_press(self, event):
        if event.button == 3:
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.fig.canvas.mpl_disconnect(self.cidrelease)
            self.fig.canvas.mpl_disconnect(self.cidmove)
            points = [circle.center for circle in self.circle_list]
            print points
            plt.close()
            return points

        if event.inaxes == self.axb3:
            if self.clase_actual == self.clase_max and self.conteo_clase_max > 0:
                self.clase_actual += 1
                self.clase_max = self.clase_actual
                self.conteo_clase_max = 0
                self.ax.set_title('Introduciendo puntos para la clase {0}'.format(self.clase_actual))
                if np.ndim(self.t) == 1:
                    self.t = np.vstack([self.t, np.zeros_like(self.t)])
                elif self.t.shape[0] <= self.clase_actual:
                    self.t = np.vstack([self.t, np.zeros_like(self.t[0])])
                self.fig.canvas.draw()

            elif self.clase_actual < self.clase_max:
                self.clase_actual += 1
                self.ax.set_title('Introduciendo puntos para la clase {0}'.format(self.clase_actual))
                self.fig.canvas.draw()
            else:
                print("No has introducido nada para la clase mas alta")
            return

        if event.inaxes == self.axb4:
            if self.clase_actual > 0:
                self.clase_actual -= 1
                self.ax.set_title('Introduciendo puntos para la clase {0}'.format(self.clase_actual))
                self.fig.canvas.draw()
            else:
                print("No existen clases negativas")
            return

        if event.inaxes == self.axb1:
            X = np.array(self.x).T
            self.metodo = LeastSquares()
            print(self.metodo.train(X, self.t))
            # Seria interesante exportar esto a un metodo suelto
            # Generar matriz de puntos, llamar a classify de self.metodo y pintarlo
            if self.clase_max == 1:
                wt = self.metodo.w_tilde.T
                rec = wt[0] - wt[1]
                x = np.arange(-20, 20)
                y = (rec[1]*x - rec[0])/-rec[2]

                self.ax.plot(x, y)
                self.fig.canvas.draw()
            elif self.clase_max == 2:
                wt = self.metodo.w_tilde.T
                rec1 = wt[0] - wt[1]
                rec2 = wt[0] - wt[2]
                rec3 = wt[1] - wt[2]
                sis = np.vstack([rec1, rec2])
                centro = np.linalg.solve(sis[:, 1:], sis[:, 0])
                xizq = np.arange(-20, centro[0], 0.25)
                xder = np.arange(centro[0], 20, 0.25)
                c1 = self.metodo.classify(np.array([20, (rec1[1] * 20 - rec1[0]) / -rec1[2]]))
                c2 = self.metodo.classify(np.array([20, (rec2[1] * 20 - rec2[0]) / -rec2[2]]))
                c3 = self.metodo.classify(np.array([20, (rec3[1] * 20 - rec3[0]) / -rec3[2]]))
                if  c1 == 0 or c1 == 1:
                    x1 = xder
                else:
                    x1 = xizq

                if c2 == 0 or c2 == 2:
                    x2 = xder
                else:
                    x2 = xizq

                if c3 == 1 or c3 == 2:
                    x3 = xder
                else:
                    x3 = xizq
                y1 = (rec1[1] * x1 - rec1[0]) / -rec1[2]
                y2 = (rec2[1] * x2 - rec2[0]) / -rec2[2]
                y3 = (rec3[1] * x3 - rec3[0]) / -rec3[2]
                self.ax.plot(x1, y1, x2, y2, x3, y3)
                self.fig.canvas.draw()
            elif self.clase_max == 3:
                wt = self.metodo.w_tilde.T
                rec1 = wt[0] - wt[1]
                rec2 = wt[0] - wt[2]
                rec3 = wt[0] - wt[3]
                rec4 = wt[1] - wt[2]
                rec5 = wt[1] - wt[3]
                rec6 = wt[2] - wt[3]
                sis = np.vstack([rec1, rec2])
                centro = np.linalg.solve(sis[:, 1:], sis[:, 0])  # Aparentemente no tiene por que existir un unico centro con 4 clases.
                xizq = np.arange(-20, centro[0], 0.25)
                xder = np.arange(centro[0], 20, 0.25)
                c1 = self.metodo.classify(np.array([20, (rec1[1] * 20 - rec1[0]) / -rec1[2]]))
                c2 = self.metodo.classify(np.array([20, (rec2[1] * 20 - rec2[0]) / -rec2[2]]))
                c3 = self.metodo.classify(np.array([20, (rec3[1] * 20 - rec3[0]) / -rec3[2]]))
                c4 = self.metodo.classify(np.array([20, (rec4[1] * 20 - rec4[0]) / -rec4[2]]))
                c5 = self.metodo.classify(np.array([20, (rec5[1] * 20 - rec5[0]) / -rec5[2]]))
                c6 = self.metodo.classify(np.array([20, (rec6[1] * 20 - rec6[0]) / -rec6[2]]))
                if c1 == 0 or c1 == 1:
                    x1 = xder
                else:
                    x1 = xizq

                if c2 == 0 or c2 == 2:
                    x2 = xder
                else:
                    x2 = xizq

                if c3 == 0 or c3 == 3:
                    x3 = xder
                else:
                    x3 = xizq

                if c4 == 1 or c4 == 2:
                    x4 = xder
                else:
                    x4 = xizq

                if c5 == 1 or c5 == 3:
                    x5 = xder
                else:
                    x5 = xizq

                if c6 == 2 or c6 == 3:
                    x6 = xder
                else:
                    x6 = xizq
                y1 = (rec1[1] * x1 - rec1[0]) / -rec1[2]
                y2 = (rec2[1] * x2 - rec2[0]) / -rec2[2]
                y3 = (rec3[1] * x3 - rec3[0]) / -rec3[2]
                y4 = (rec4[1] * x4 - rec4[0]) / -rec4[2]
                y5 = (rec5[1] * x5 - rec5[0]) / -rec5[2]
                y6 = (rec6[1] * x6 - rec6[0]) / -rec6[2]

                self.ax.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6)
                self.fig.canvas.draw()
            clase_fondo = self.metodo.classify(self.fondo)
            print(clase_fondo)
            return

        if event.inaxes == self.axb2:
            X = np.array(self.x).T
            self.metodo = LDA()
            print(self.metodo.train(X, self.t))
            #por ahora no podemos clasificar
            return

        x0, y0 = event.xdata, event.ydata
        for circle in self.circle_list:
            contains, attr = circle.contains(event)
            if contains:
                self.press_event = event
                self.current_circle = circle
                self.x0, self.y0 = self.current_circle.center
                self.current_x = self.x.index(self.current_circle.center)
                return

        c = Circle((x0, y0), 0.5, color='C{0}'.format(self.clase_actual))
        self.ax.add_patch(c)
        self.circle_list.append(c)
        self.x.append((x0, y0))

        if self.clase_actual == 0 and self.clase_max == 0:
            self.t = np.hstack(([self.t, 1]))
        else:
            vec_can = np.zeros(self.clase_max + 1)
            vec_can[self.clase_actual] = 1
            self.t = np.hstack([self.t, vec_can.reshape(vec_can.shape[0], 1)])
        if self.clase_actual == self.clase_max:
            self.conteo_clase_max += 1

        self.current_circle = None
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.current_circle is not None:
            x0, y0 = event.xdata, event.ydata
            self.x[self.current_x] = (x0, y0)
        if self.metodo is not None and self.current_circle is not None:
            X = np.array(self.x).T
            print(self.metodo.train(X, self.t))
            # Dependiendo del metodo habra que llamar a un pintar o a otro

        self.current_x = None
        self.press_event = None
        self.current_circle = None

    def on_move(self, event):
        if (self.press_event is None or
            event.inaxes != self.press_event.inaxes or
            self.current_circle == None):
            return
        
        dx = event.xdata - self.press_event.xdata
        dy = event.ydata - self.press_event.ydata
        self.current_circle.center = self.x0 + dx, self.y0 + dy
        self.fig.canvas.draw()


if __name__ == '__main__':

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    axb1 = plt.axes([0.6, 0.05, 0.2, 0.075])
    axb2 = plt.axes([0.81, 0.05, 0.1, 0.075])
    axb3 = plt.axes([0.1, 0.05, 0.1, 0.075])
    axb4 = plt.axes([0.21, 0.05, 0.1, 0.075])
    
    start = CreatePoints(fig, ax, axb1, axb2, axb3, axb4)
    plt.show()
