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

        self.rectas_a_borrar = 0

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

        elif event.inaxes == self.axb4:
            if self.clase_actual > 0:
                self.clase_actual -= 1
                self.ax.set_title('Introduciendo puntos para la clase {0}'.format(self.clase_actual))
                self.fig.canvas.draw()
            else:
                print("No existen clases negativas")
            return

        elif event.inaxes == self.axb1:
            if self.clase_max > 0:
                if self.rectas_a_borrar > 0:
                    self.borrar_rectas()
                X = np.array(self.x).T
                self.metodo = LeastSquares()
                print(self.metodo.train(X, self.t))
                if self.clase_max == 1:
                    wt = self.metodo.w_tilde.T
                    rec = wt[0] - wt[1]
                    x = np.arange(-20, 20)
                    y = (rec[1]*x - rec[0])/-rec[2]
                    self.ax.plot(x, y)
                    self.rectas_a_borrar += 1
                    self.fig.canvas.draw()
                elif self.clase_max >= 2:
                    wt = self.metodo.w_tilde.T
                    rec1 = wt[0] - wt[1]
                    rec2 = wt[0] - wt[2]
                    sis = np.vstack([rec1, rec2])
                    centro = np.linalg.solve(sis[:, 1:], sis[:, 0])
                    for i in range(0, self.clase_max+1):
                        for j in range(i+1, self.clase_max+1):
                            rec = wt[i] - wt[j]
                            self.draw_line(rec, centro, (i, j))
                    self.fig.canvas.draw()

                clase_fondo = self.metodo.classify(self.fondo)
                print(clase_fondo)
            else:
                print("Como vas a clasificar si solo tienes una clase, Sherlock")
            return

        elif event.inaxes == self.axb2:
            X = np.array(self.x).T
            self.metodo = LDA()
            print(self.metodo.train(X, self.t))
            #por ahora no podemos clasificar
            return

        else:
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

    def draw_line(self, eqn, cnt, clases):
        clase = self.metodo.classify(np.array([20, (eqn[1] * 20 - eqn[0]) / -eqn[2]]))
        if clase == clases[0] or clase == clases[1]:
            x = np.arange(cnt[0], 20, 0.25)
        else:
            x = np.arange(-20, cnt[0], 0.25)
        rec = (eqn[1] * x - eqn[0]) / -eqn[2]
        self.rectas_a_borrar += 1
        self.ax.plot(x, rec)

    def borrar_rectas(self):
        for i in range(self.rectas_a_borrar):
            self.ax.lines.pop(0)
        self.rectas_a_borrar = 0

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
