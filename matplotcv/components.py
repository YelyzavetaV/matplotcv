import os.path
from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.graphics import Color, Line
from kivy.properties import ObjectProperty
from utils import point_segment_distance

Builder.load_file('components.kv')


def print_widget_hierarchy(widget, level=0):
    print("  " * level + f"{widget.__class__.__name__}: {widget}")
    for child in widget.children:
        print_widget_hierarchy(child, level + 1)


class TransparentScatter(Scatter):

    def on_touch_down(self, touch):
        # Prevent Scatter from bringing itself to the front
        if self.collide_point(*touch.pos):
            return super(Scatter, self).on_touch_down(touch)
        return False


class HoverButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._hovered = False
        self.background_hovered = kwargs.get(
            'background_hovered', self.background_normal
        )

        Window.bind(mouse_pos=self.on_mouse_move)

    def on_kv_post(self, widget):
        self._background_normal = self.background_normal
        return super().on_kv_post(widget)

    @property
    def hovered(self):
        return self._hovered

    def on_mouse_move(self, window, pos):
        if self.collide_point(*self.to_widget(*pos)):
            if not self.hovered:
                self.on_enter()
        else:
            if self.hovered:
                self.on_leave()

    def on_enter(self):
        self._hovered = True
        self.background_normal = self.background_hovered

    def on_leave(self):
        self._hovered = False
        self.background_normal = self._background_normal


class FileChooserContent(BoxLayout):
    file_chooser = ObjectProperty()
    load = ObjectProperty()
    cancel = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_chooser.path = kwargs.get('path', os.path.expanduser('~'))


def open_nested_dropdown(dropdown, button):
    dropdown.open(button)
    pos = button.to_window(button.x, button.y, relative=True)
    dropdown.pos = (pos[0] + button.width, pos[1])


class ToolsDropDown(DropDown):
    blur_dropdown = ObjectProperty()
    detect_edges_dropdown = ObjectProperty()
    open_nested_dropdown = staticmethod(open_nested_dropdown)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blur_dropdown = Factory.BlurDropDown()
        self.detect_edges_dropdown = Factory.DetectEdgesDropDown()


class ContourDropDown(DropDown):
    pass


class ContourWidget(Widget):
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
            if point_segment_distance(
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