import cv2 as cv

import kivy
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger, LOG_LEVELS

import exceptions
from pipeline import Pipeline
from components import (
    FileChooserContent, ResizeDropDown, ToolsDropDown, DrawDropDown
)
from contour import Contour

kivy.require('2.3.0')
Logger.setLevel(LOG_LEVELS['debug'])


class MPLWidget(Widget):
    app = ObjectProperty()
    layout = ObjectProperty()
    scatter = ObjectProperty()
    image = ObjectProperty()
    original_image_toggle = ObjectProperty()
    pipeline = Pipeline()
    contours = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Window.bind(on_resize=self.on_window_resize)

        self.reduce_dropdown = ResizeDropDown()
        self.reduce_dropdown.bind(
            on_select=lambda i, v: self.pipeline.resize(v)
        )

        self.tools_dropdown = ToolsDropDown()
        self.tools_dropdown.bind(on_select=lambda i, v: self.pipeline.gray())
        self.tools_dropdown.blur_dropdown.bind(
            on_select=lambda i, v: self.pipeline.blur(v)
        )
        self.tools_dropdown.detect_edges_dropdown.bind(
            on_select=lambda i, v: self.pipeline.edges(v)
        )

        self.draw_dropdown = DrawDropDown()
        self.draw_dropdown.bind(
            on_select=lambda i, v: self.draw_contours()
        )
        self.draw_dropdown.bind(
            on_select=lambda i, v: self.clear_contours()
        )

    #---------------------------
    # UI operations
    #---------------------------
    def on_window_resize(self, instance, w, h):
        if self.contours:
            Clock.unschedule(self.draw_contours)  # Prevent multiple calls
            Clock.schedule_once(lambda dt: self.draw_contours(), 0.1)

    def on_load_image_button_press(self):
        content = FileChooserContent(
            load=self.load_image_with_file_chooser,
            cancel=self.dismiss_file_chooser,
        )
        self.file_chooser = Popup(
            content=content,
            title='Load Image',
            # background='',
        )
        self.file_chooser.open()

    def load_image_with_file_chooser(self, selection):
        if selection:  # Try loading file once confirmed with load button
            if self.file_chooser.content.load_button.state == 'down':
                try:
                    self.clear()

                    self.pipeline.load_image(selection[0])

                    if self.pipeline.empty:
                        raise exceptions.ImageLoadError()

                    self.update_image()

                    # Sync at 30 FPS
                    Clock.schedule_interval(
                        lambda interval: self.update_image(), 1 / 30
                    )
                except Exception as e:
                    raise e
                finally:  # Cleanup
                    self.dismiss_file_chooser()

    def dismiss_file_chooser(self):
        self.file_chooser.dismiss()

    def on_original_image_toggle_press(self):
        if self.original_image_toggle.state == 'down':
            self.app.config.set('General', 'show_pipeline', 'On')
        else:
            self.app.config.set('General', 'show_pipeline', 'Off')
        self.app.config.write()

    #---------------------------
    # Image operations
    #---------------------------
    def resize_image(self):
        w, h = self.size
        aspect = self.pipeline.aspect

        if w / h > aspect:
            self.image.size = (h * aspect, h)
        else:
            self.image.size = (w, w / aspect)

    def update_image(self):
        if not self.pipeline.empty:
            show_pipeline = self.app.config.get(
                'General', 'show_pipeline'
            ) == 'On'
            image = (
                self.pipeline.image
                if show_pipeline else self.pipeline.original
            )

            colorfmt = 'luminance' if image.ndim == 2 else 'bgr'

            buff = cv.flip(image, 0).tobytes()
            texture = Texture.create(
                size=(image.shape[1], image.shape[0]), colorfmt=colorfmt
            )
            texture.blit_buffer(buff, colorfmt=colorfmt, bufferfmt='ubyte')

            self.image.texture = texture
            self.resize_image()
            self.center_image()

    def center_image(self):
        if self.image.texture:
            self.image.pos = (
                0.5 * (self.width - self.image.width),
                0.5 * (self.height - self.image.height),
            )

    def zoom(self, factor):
        self.scatter.scale *= factor

    def draw_contours(self):
        '''Draw OpenCV contours as widgets'''
        if not self.pipeline.empty:
            p = self.pipeline

            if not p.edges_detected:
                p.edges()
            if not p.contours:
                p.contour_tree()

            # Map contours to widget coordinates
            w, h = self.image.size
            x, y = self.image.pos

            scale = (
                w / self.pipeline.original.shape[1],
                h / self.pipeline.original.shape[0],
            )

            if self.contours:
                self.clear_contours()

            for c in p.contours:
                mapped = [
                    (
                        x + p[0][0] * scale[0], y + h - p[0][1] * scale[1]
                    ) for p in c
                ]
                contour = Contour(mapped)
                self.image.add_widget(contour)
                self.contours.append(contour)

    def clear_contours(self):
        for contour in self.contours:
            self.image.remove_widget(contour)
        self.contours = []

    def clear(self):
        self.pipeline.clear('all')
        self.clear_contours()
        self.image.texture = None


class MPLApp(App):
    title = 'MPLCV'

    def build(self):
        Window.minimum_width = self.config.getint('Graphics', 'min_width')
        Window.minimum_height = self.config.getint('Graphics', 'min_height')

        self.mpl_widget = MPLWidget(app=self)
        return self.mpl_widget

    def build_config(self, config):
        config.adddefaultsection('General')
        config.setdefault('General', 'show_pipeline', 'Off')

        config.adddefaultsection('Graphics')
        config.setdefault('Graphics', 'min_width', 800)
        config.setdefault('Graphics', 'min_height', 600)

    def build_settings(self, settings):
        settings.add_json_panel(
            'General',
            self.config,
            data='''
            [
                {'type': 'title', 'title': 'General'},
                {
                'type': 'bool',
                'title': 'Show pipeline',
                'desc': 'Show/hide the pipeline modifications',
                'section': 'General',
                'key': 'show_pipeline',
                'values': ['On', 'Off']
                },
            ]
            '''
        )

        settings.add_json_panel(
            'Graphics',
            self.config,
            data='''
            [
                {'type': 'title', 'title': 'Graphics'},
            ]
            '''
        )


if __name__ == '__main__':
    MPLApp().run()
