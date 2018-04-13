import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider, RadioButtons
from Perceptron import *


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

    def __init__(self, fig, ax, bperc, siters, seta, radio, ejes):
        self.circle_list = []
        self.t = np.array([])
        self.conteo_clase = [0, 0]
        self.clase_actual = 0
        self.eta = 0.1
        self.iters = 3
        self.clases = {'Clase 1': 0, 'Clase 2': 1}

        self.perceptron = None
        self.x0 = None
        self.y0 = None
        self.ejes = ejes

        self.scat_rem = None
        self.rec = None

        self.fig = fig
        self.ax = ax
        self.fondo = np.mgrid[-20:20:0.5, -20:20:0.5].reshape(2, 6400)

        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        siters.on_changed(self.update)
        seta.on_changed(self.update)
        bperc.on_clicked(self.ejec)
        radio.on_clicked(self.colorfunc)

        self.press_event = None
        self.current_circle = None

    def colorfunc(self, label):
        self.clase_actual = self.clases[label]

    def ejec(self, event):
        if self.conteo_clase[0] == 0 or self.conteo_clase[1] == 0:
            print('No has rellenado dos clases')
        else:
            if self.perceptron is None:
                self.perceptron = Perceptron(2, int(np.rint(self.iters)))
            self.perceptron.iters = self.iters
            x = self.parsea_circulos()
            self.perceptron.train(x, self.t, eta=self.eta)
            self.colorize_bg()

    def update(self, val):
        self.eta = seta.val
        self.iters = siters.val

    def colorize_bg(self):
        if self.scat_rem is not None:
            self.scat_rem.remove()
            self.scat_rem = None

        clase_fondo = map((lambda x: 'C0' if self.perceptron.eval_weights(x) > 0 else 'C1'), self.fondo.T)
        self.scat_rem = self.ax.scatter(self.fondo[0], self.fondo[1], color=clase_fondo, alpha=0.2, s=5)

        self.fig.canvas.draw()

    def parsea_circulos(self):
        x = []
        for circle in self.circle_list:
            x.append(circle.center)
        return np.array(x)

    def on_press(self, event):
        if event.button == 3:  # Pulsar con el boton derecho
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.fig.canvas.mpl_disconnect(self.cidrelease)
            self.fig.canvas.mpl_disconnect(self.cidmove)
            points = [circle.center for circle in self.circle_list]
            print points
            plt.close()
            return points

        elif event.inaxes != self.ejes[0] and \
             event.inaxes != self.ejes[1] and \
             event.inaxes != self.ejes[2] and \
             event.inaxes != self.ejes[3]:  # Pulsar dentro del grafico
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

            if self.clase_actual == 0:
                self.t = np.hstack([self.t, 1])
                self.conteo_clase[0] += 1
            else:
                self.t = np.hstack([self.t, -1])
                self.conteo_clase[1] += 1

            self.current_circle = None
            self.fig.canvas.draw()

    def on_release(self, event):
        # A lo mejor hay que meter aqui que el evento sea dentro de la figura para evitar el bug del readme
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
    plt.subplots_adjust(left=0.25, bottom=0.25)

    axiters = plt.axes([0.25, 0.1, 0.65, 0.03])
    axeta = plt.axes([0.25, 0.15, 0.65, 0.03])
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    rax = plt.axes([0.025, 0.5, 0.15, 0.15])
    bperc = Button(resetax, 'Perceptron', hovercolor='0.975')
    f0 = 3
    siters = Slider(axiters, 'Iters', 1, 10, valinit=f0)
    a0 = 0.1
    seta = Slider(axeta, 'Eta', 0.01, 1, valinit=a0)
    radio = RadioButtons(rax, ('Clase 1', 'Clase 2'), active=0)
    ejes = [axiters, axeta, resetax, rax]

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')

    start = CreatePoints(fig, ax, bperc, siters, seta, radio, ejes)
    plt.show()
