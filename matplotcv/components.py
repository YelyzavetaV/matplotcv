import os.path
import matplotlib.colors as colors

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.scatter import Scatter
from kivy.uix.dropdown import DropDown
from kivy.graphics import Color, Line
from kivy.properties import ObjectProperty, StringProperty

from metrics import point_segment_distance

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


class MessagePopup(Popup):
    title = StringProperty('')
    message = StringProperty('')


class ErrorPopup(MessagePopup):

    def on_kv_post(self, base_widget):
        self.title = 'Something went wrong'
        return super().on_kv_post(base_widget)


class ConfirmationPopup(Popup):
    message = StringProperty('')
    confirm = ObjectProperty()


class FinderPopup(Popup):
    icon_view = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.icon_view.path = kwargs.get('path', os.path.expanduser('~'))


class FileLoadPopup(FinderPopup):
    load = ObjectProperty()


class FileSavePopup(FinderPopup):
    save = ObjectProperty()


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


def open_nested_dropdown(dropdown, button, parent):
    '''
    Open a nested dropdown and align it with the parent at the bottom.
    '''
    dropdown.open(button)
    pos = button.to_window(button.x, button.y, relative=True)
    dropdown.pos = (
        pos[0] + button.width, pos[1] + button.height - dropdown.height
    )

    dropdown.bind(on_dismiss=lambda i: parent.dismiss())


class BaseDropDown(DropDown):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def open(self, widget, pos):
        super().open(widget)

        x, y = pos

        if x + self.width > Window.width:
            x = Window.width - self.width

        if y < 0:
            y = 0
        if y + self.height > Window.height:
            y = Window.height - self.height

        self.pos = (x, y)


class ToolsDropDown(DropDown):
    blur_dropdown = ObjectProperty()
    detect_edges_dropdown = ObjectProperty()
    open_nested_dropdown = staticmethod(open_nested_dropdown)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blur_dropdown = Factory.BlurDropDown()
        self.detect_edges_dropdown = Factory.DetectEdgesDropDown()


class MathDropDown(DropDown):
    log_scale_dropdown = ObjectProperty()
    open_nested_dropdown = staticmethod(open_nested_dropdown)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_scale_dropdown = LogScaleDropDown()


class LogScaleDropDown(DropDown):
    selection = StringProperty('OFF')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bind(on_select=self.update)

    def open(self, *args):
        for child in self.container.children:
            child.state = 'down' if child.text == self.selection else 'normal'
        super().open(*args)

    def update(self, instance, value):
        self.selection = value

        app = App.get_running_app()
        app.config.set('Math', 'log_scale', value)
        app.config.write()


class ContourDropDown(BaseDropDown):
    label_axis_dropdown = ObjectProperty()
    open_nested_dropdown = staticmethod(open_nested_dropdown)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_axis_dropdown = Factory.LabelAxisDropDown()


class ContourWidget(Widget):
    '''
    Handles the logic of interactable contours. Stores the unique key
    of the contour to identify it in the pipeline.
    '''

    def __init__(self, key, points, color='blue', **kwargs):
        super().__init__(**kwargs)

        self._hovered = False
        self.key = key
        self.color = colors.to_rgba(color)

        self.update(points)

        self.contour_dropdown = ContourDropDown()

        # Delegate contour actions to MPLWidget
        root_widget = App.get_running_app().root_widget
        actions = {
            'Split':
            lambda instance: root_widget.split_contour(self.key),
            'Clear':
            lambda instance: root_widget.clear_contour(self.key),
            'Label as...':
            lambda instance: self.contour_dropdown.open_nested_dropdown(
                self.contour_dropdown.label_axis_dropdown,
                instance,
                self.contour_dropdown,
            ),
            'Export...':
            self.on_export_button_press,
        }
        for action, callback in actions.items():
            button = Button(text=action, height=50, size_hint_y=None)

            button.bind(on_press=callback)
            button.bind(on_release=self.contour_dropdown.dismiss)

            self.contour_dropdown.add_widget(button)

        self.contour_dropdown.label_axis_dropdown.bind(
            on_select=self.on_label_button_press
        )

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
                Color(*self.color)

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
            self.contour_dropdown.open(self, touch.pos)
            return True
        return super().on_touch_down(touch)

    def on_label_button_press(self, instance, value):
        root_widget = App.get_running_app().root_widget

        match value:
            case 'tick':
                input_popup = Factory.TickInputPopup()
                input_popup.on_submit = lambda v: root_widget.label_contour(
                    self.key, label=value, coordinate=v
                )
                input_popup.open()
            case _:
                root_widget.label_contour(self.key, label=value)

    def on_export_button_press(self, instance):
        root_widget = App.get_running_app().root_widget
        root_widget.marked_contours.add(self.key)

        root_widget.file_saver.open()
