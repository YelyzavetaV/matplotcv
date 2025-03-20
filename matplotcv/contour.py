import math
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import ObjectProperty


class Contour(Widget):
    pos = ObjectProperty((0, 0))
    size = ObjectProperty((0, 0))

    def __init__(self, points, **kwargs):
        super().__init__(**kwargs)

        self.selected = False
        self._hovered = False

        self.update(points)

    @property
    def hovered(self):
        return self._hovered

    @hovered.setter
    def hovered(self, value):
        if self.hovered != value:
            self._hovered = value
            print(f"Hover state changed to: {value}")
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
            p1 = self.points[i]
            p2 = self.points[i + 1]

            if self.point_to_segment_distance((x, y), p1, p2) < threshold:
                return True
        return False

    @staticmethod
    def point_to_segment_distance(point, start, end):
        x, y = point
        sx, sy = start
        ex, ey = end

        length = (ex - sx)**2 + (ey - sy)**2
        if length == 0:  # Segment's start and end points are the same
            return math.sqrt((x - sx)**2 + (y - sy)**2)

        # Calculate the projection of the point onto the line
        tau = max(
            0, min(1, ((x - sx) * (ex - sx) + (y - sy) * (ey - sy)) / length)
        )
        proj = (
            sx + tau * (ex - sx), sy + tau * (ey - sy)
        )

        return math.sqrt((x - proj[0])**2 + (y - proj[1])**2)
