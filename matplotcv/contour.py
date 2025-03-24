import math
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Line
from components import ContourDropDown


def _point_segment_distance(
    point: tuple[float], segment: tuple[tuple[float]]
) -> float:
    '''Calculate the distance between a point and a line segment.'''
    x, y = point
    sx, sy = segment[0]
    ex, ey = segment[1]

    length = (ex - sx)**2 + (ey - sy)**2
    if length == 0:  # Segment's start and end points are the same
        return math.sqrt((x - sx)**2 + (y - sy)**2)

    # Calculate the projection of the point onto the line
    tau = max(
        0, min(1, ((x - sx) * (ex - sx) + (y - sy) * (ey - sy)) / length)
    )
    proj = (sx + tau * (ex - sx), sy + tau * (ey - sy))

    return math.sqrt((x - proj[0])**2 + (y - proj[1])**2)


class Contour(Widget):
    '''Handles interactable contours.'''

    def __init__(self, key, points, **kwargs):
        super().__init__(**kwargs)

        self.selected = False
        self._hovered = False
        self.key = key

        self.update(points)

        self.dropdown = ContourDropDown()

        # Delegate contour actions to MPLWidget
        mpl_widget = App.get_running_app().mpl_widget
        actions = {
            'Split': lambda i: mpl_widget.split_contour(self.key),
            'Clear': lambda i: mpl_widget.clear_contour(self.key),
        }
        for action, callback in actions.items():
            button = Button(text=action, height=50, size_hint_y=None)
            button.bind(on_press=callback, on_release=self.dropdown.dismiss)
            self.dropdown.add_widget(button)

    @property
    def hovered(self):
        return self._hovered

    @hovered.setter
    def hovered(self, value):
        if self.hovered != value:
            self._hovered = value
            self.update(self.points)

    def update(self, points):
        self.points = points

        self.canvas.clear()
        with self.canvas:
            if self.hovered:
                Color(0, 1, 0, 1)
            else:
                Color(0, 0, 1, 1)

            Line(points=sum(points, ()), width=2)

    def collide_point(self, x, y, threshold=10):
        '''
        For better detection of collisions when contour's points are far
        apart, we calculate the distance from the point to each segment
        of the contour.
        '''
        for i in range(len(self.points) - 1):
            if _point_segment_distance(
                (x, y), (self.points[i], self.points[i + 1])
            ) < threshold:
                return True
        return False

    def on_touch_down(self, touch):
        if touch.button == 'left' and self.collide_point(*touch.pos):
            self.dropdown.open(self)
            self.dropdown.pos = touch.pos
            return True
        return super().on_touch_down(touch)
