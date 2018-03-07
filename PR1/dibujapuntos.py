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
        self.t = np.array([])
        self.conteo_clase_max = 0
        self.clase_max = 0
        self.clase_actual = 0

        self.metodo = None
        self.x0 = None
        self.y0 = None

        self.fig = fig
        self.ax = ax
        self.fondo = np.mgrid[-20:20:0.5, -20:20:0.5].reshape(2, 6400)
        
        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        self.press_event = None
        self.current_circle = None
        self.scat_rem = None

        self.axb1 = axb1
        Button(self.axb1, 'Least Squares')
        self.axb2 = axb2
        Button(self.axb2, 'LDA')
        self.axb3 = axb3
        Button(self.axb3, 'Clase +')
        self.axb4 = axb4
        Button(self.axb4, 'Clase -')

        self.ax.set_title('Introduciendo puntos para la clase 0')

    def colorize_bg(self):
        if self.scat_rem is not None:
            self.scat_rem.remove()
            self.scat_rem = None
        clase_fondo = map((lambda x: 'C' + str(x)), self.metodo.classify(self.fondo))
        self.scat_rem = self.ax.scatter(self.fondo[0], self.fondo[1], color=clase_fondo, alpha=0.2, s=5)
        self.fig.canvas.draw()

    def actualiza_titulo(self):
        self.ax.set_title('Introduciendo puntos para la clase {0}'.format(self.clase_actual))
        self.fig.canvas.draw()

    def parsea_circulos(self):
        x = []
        for circle in self.circle_list:
            x.append(circle.center)
        return np.array(x).T

    def on_press(self, event):
        if event.button == 3:                  # Pulsar con el boton derecho
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.fig.canvas.mpl_disconnect(self.cidrelease)
            self.fig.canvas.mpl_disconnect(self.cidmove)
            points = [circle.center for circle in self.circle_list]
            print points
            plt.close()
            return points

        if event.inaxes == self.axb3:           # Pulsar en Clase +
            if self.clase_actual == self.clase_max and self.conteo_clase_max > 0:
                self.clase_actual += 1
                self.clase_max = self.clase_actual
                self.conteo_clase_max = 0
                if np.ndim(self.t) == 1:
                    self.t = np.vstack([self.t, np.zeros_like(self.t)])
                elif self.t.shape[0] <= self.clase_actual:
                    self.t = np.vstack([self.t, np.zeros_like(self.t[0])])
                self.actualiza_titulo()

            elif self.clase_actual < self.clase_max:
                self.clase_actual += 1
                self.actualiza_titulo()

            else:
                print("No has introducido nada para la clase mas alta")
            return

        elif event.inaxes == self.axb4:         # Pulsar en Clase -
            if self.clase_actual > 0:
                self.clase_actual -= 1
                self.actualiza_titulo()
            else:
                print("No existen clases negativas")
            return

        elif event.inaxes == self.axb1:         # Pulsar en Least Squares
            if self.clase_max > 0:
                x = self.parsea_circulos()
                self.metodo = LeastSquares()
                print(self.metodo.train(x, self.t))
                self.colorize_bg()

            else:
                print("Como vas a clasificar si solo tienes una clase, Sherlock")
            return

        elif event.inaxes == self.axb2:         # Pulsar en LDA
            x = self.parsea_circulos()
            self.metodo = LDA()
            print(self.metodo.train(x, self.t))
            # por ahora no podemos clasificar
            return

        else:                                   # Pulsar dentro del grafico
            x0, y0 = event.xdata, event.ydata
            for circle in self.circle_list:
                contains, attr = circle.contains(event)
                if contains:
                    self.press_event = event
                    self.current_circle = circle
                    self.x0, self.y0 = self.current_circle.center
                    return

            c = Circle((x0, y0), 0.5, color='C{0}'.format(self.clase_actual))
            self.ax.add_patch(c)
            self.circle_list.append(c)

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
        if self.metodo is not None and self.current_circle is not None:
            x = self.parsea_circulos()
            print(self.metodo.train(x, self.t))
            self.colorize_bg()
        self.press_event = None
        self.current_circle = None

    def on_move(self, event):
        if self.press_event is None or \
                event.inaxes != self.press_event.inaxes or \
                self.current_circle is None:
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
