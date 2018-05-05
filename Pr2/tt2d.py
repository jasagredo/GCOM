import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, TextBox
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

    def __init__(self, fig, ax, widgets):
        self.circle_list = []
        self.t = np.array([])
        self.conteo_clase = [0, 0]
        self.clase_actual = 0
        self.eta = 0.1
        self.iters = 3
        self.clases = {'Clase 1': 0, 'Clase 2': 1}
        self.w0 = None

        self.perceptron = None

        self.scat_rem = None
        self.fondo = np.mgrid[-20:20:0.5, -20:20:0.5].reshape(2, 6400)

        self.fig = fig
        self.ax = ax

        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        widgets['radio'].on_clicked(self.selec)

        widgets['w0x'].on_submit(self.upd_w0x)
        widgets['w0y'].on_submit(self.upd_w0y)
        widgets['bias'].on_submit(self.upd_bias)
        widgets['eta'].on_submit(self.upd_eta)
        widgets['iters'].on_submit(self.upd_iters)

        widgets['per'].on_clicked(self.ejec)

        self.x0 = None
        self.y0 = None
        self.press_event = None
        self.current_circle = None

    def selec(self, label):
        self.clase_actual = self.clases[label]

    def upd_w0x(self, text):
        if self.w0 is None:
            self.w0 = np.zeros(3)
        self.w0[1] = float(text)

    def upd_w0y(self, text):
        if self.w0 is None:
            self.w0 = np.zeros(3)
        self.w0[2] = float(text)

    def upd_bias(self, text):
        if self.w0 is None:
            self.w0 = np.zeros(3)
        self.w0[0] = float(text)

    def upd_eta(self, text):
        self.eta = float(text)

    def upd_iters(self, text):
        self.iters = int(np.rint(float(text)))

    def ejec(self, event):
        if self.conteo_clase[0] == 0 or self.conteo_clase[1] == 0:
            print('No has rellenado dos clases')
        else:
            if self.perceptron is None:
                self.perceptron = Perceptron(2, self.iters)
            else:
                self.perceptron.iters = self.iters
            x = self.parsea_circulos()
            self.perceptron.train(x, self.t, w0=self.w0, eta=self.eta)
            self.colorize_bg()

    def colorize_bg(self):
        if self.scat_rem is not None:
            self.scat_rem.remove()
            self.scat_rem = None

        clase_fondo = map((lambda x: 'C0' if self.perceptron.eval(x) > 0 else 'C1'), self.fondo.T)
        self.scat_rem = self.ax.scatter(self.fondo[0], self.fondo[1], color=clase_fondo, alpha=0.2, s=5)

        self.fig.canvas.draw()

    def parsea_circulos(self):
        x = []
        for circle in self.circle_list:
            x.append(circle.center)
        return np.array(x).T

    def on_press(self, event):
        if event.button == 3:
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.fig.canvas.mpl_disconnect(self.cidrelease)
            self.fig.canvas.mpl_disconnect(self.cidmove)
            points = [circle.center for circle in self.circle_list]
            print points
            plt.close()
            return points

        elif event.inaxes == self.ax:
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

    resetax = plt.axes([0.025, 0.1, 0.25, 0.04])
    rax = plt.axes([0.1, 0.7, 0.15, 0.15])
    bperc = Button(resetax, 'Perceptron', hovercolor='0.975')
    radio = RadioButtons(rax, ('Clase 1', 'Clase 2'), active=0)

    w0x_ini = '0.0'
    axbox = plt.axes([0.1, 0.6, 0.15, 0.04])
    w0x = TextBox(axbox, 'w0[x]', initial=w0x_ini)

    w0y_ini = '0.0'
    axbox = plt.axes([0.1, 0.55, 0.15, 0.04])
    w0y = TextBox(axbox, 'w0[y]', initial=w0y_ini)

    bias_ini = '0.0'
    axbox = plt.axes([0.1, 0.5, 0.15, 0.04])
    bias = TextBox(axbox, 'Bias', initial=bias_ini)

    eta_ini = '0.1'
    axbox = plt.axes([0.1, 0.4, 0.15, 0.04])
    eta = TextBox(axbox, 'Eta', initial=eta_ini)

    iters_ini = '3'
    axbox = plt.axes([0.1, 0.3, 0.15, 0.04])
    iters = TextBox(axbox, 'Iters', initial=iters_ini)

    widgets = {'eta': eta, 'iters': iters, 'bias': bias, 'w0x': w0x, 'w0y': w0y, 'per': bperc, 'radio': radio}

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')

    start = CreatePoints(fig, ax, widgets)
    plt.show()
