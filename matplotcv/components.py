import os.path
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.properties import ObjectProperty

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
