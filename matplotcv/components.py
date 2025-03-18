import os.path
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.properties import ObjectProperty


class Background(Image):
    pass


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


class ResizeDropDown(DropDown):
    pass


class ToolsDropDown(DropDown):
    blur_dropdown = ObjectProperty()
    detect_edges_dropdown = ObjectProperty()
    open_nested_dropdown = staticmethod(open_nested_dropdown)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blur_dropdown = BlurDropDown()
        self.detect_edges_dropdown = DetectEdgesDropDown()


class BlurDropDown(DropDown):
    pass


class DetectEdgesDropDown(DropDown):
    pass


class DrawDropDown(DropDown):
    draw_contours_dropdown = ObjectProperty()
    clear_contours_dropdown = ObjectProperty()
    open_nested_dropdown = staticmethod(open_nested_dropdown)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draw_contours_dropdown = DrawContoursDropDown()
        self.clear_contours_dropdown = ClearContoursDropDown()


class DrawContoursDropDown(DropDown):
    pass


class ClearContoursDropDown(DropDown):
    pass