from kivy.uix.widget import Widget
from kivy.graphics import Color, Line


class Contour(Widget):

    def __init__(self, points, **kwargs):
        super().__init__(**kwargs)
        self.selected = False
        self.update(points)

    def update(self, points):
        self.points = points
        self.canvas.clear()
        with self.canvas:
            Color(0, 0, 1)
            Line(points=sum(points, ()), width=2)
