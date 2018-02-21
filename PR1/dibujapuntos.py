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

    def redraw(self):
        self.circle_list = []
        self.fig.canvas.draw()
    
    def __init__(self, fig, ax, axb1, axb2, axb3):
        self.circle_list = []

        self.x0 = None
        self.y0 = None

        self.fig = fig
        self.ax = ax
        
        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        self.press_event = None
        self.current_circle = None

        self.axb1 = axb1
        self.axb2 = axb2
        self.axb3 = axb3
        Button(self.axb1, 'Least Squares')
        Button(self.axb2, 'LDA')
        self.bclases = Button(self.axb3, '0')

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
            return

        if event.inaxes == self.axb1:
            print("ls")
            return
        if event.inaxes == self.axb2:
            print("lda")
            return

        x0, y0 = event.xdata, event.ydata
        for circle in self.circle_list:
            contains, attr = circle.contains(event)
            if contains:
                self.press_event = event
                self.current_circle = circle
                self.x0, self.y0 = self.current_circle.center
                return

        c = Circle((x0, y0), 0.5)
        self.ax.add_patch(c)
        self.circle_list.append(c)
        self.current_circle = None
        self.fig.canvas.draw()

    def on_release(self, event):
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
    
    start = CreatePoints(fig, ax, axb1, axb2, axb3)
    plt.show()
