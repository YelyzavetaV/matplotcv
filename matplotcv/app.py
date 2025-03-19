import cv2 as cv

import kivy
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

kivy.require('2.3.0')
Logger.setLevel(LOG_LEVELS['debug'])


class MPLWidget(Widget):
    app = ObjectProperty()
    layout = ObjectProperty()
    scatter = ObjectProperty()
    image = ObjectProperty()
    pipelines = []  # Store active opencv states
    active_pipeline = None
    original_image_toggle = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reduce_dropdown = ResizeDropDown()
        self.reduce_dropdown.bind(on_select=self.resize_image)

        self.tools_dropdown = ToolsDropDown()
        self.tools_dropdown.bind(on_select=self.gray_image)
        self.tools_dropdown.blur_dropdown.bind(on_select=self.blur_image)
        self.tools_dropdown.detect_edges_dropdown.bind(
            on_select=self.detect_edges
        )

        self.draw_dropdown = DrawDropDown()
        self.draw_dropdown.draw_contours_dropdown.bind(
            on_select=self.draw_contours
        )
        self.draw_dropdown.clear_contours_dropdown.bind(
            on_select=self.clear_contours
        )

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

    def on_original_image_toggle_press(self):
        if self.original_image_toggle.state == 'down':
            self.app.config.set('General', 'show_pipeline', 'On')
        else:
            self.app.config.set('General', 'show_pipeline', 'Off')
        self.app.config.write()

    def load_image_with_file_chooser(self, selection):
        if selection:  # Try loading file once confirmed with load button
            if self.file_chooser.content.load_button.state == 'down':
                try:
                    pipeline = Pipeline.from_file(selection[0])

                    if pipeline is None:
                        raise exceptions.ImageLoadError()

                    self.active_pipeline = pipeline

                    self.pipelines.append(self.active_pipeline)
                    self.update_image()

                    # Sync at 30 FPS
                    Clock.schedule_interval(self.sync_texture, 1 / 30)
                except Exception as e:
                    raise e
                finally:  # Cleanup
                    self.dismiss_file_chooser()

    def dismiss_file_chooser(self):
        self.file_chooser.dismiss()

    def update_image(self):
        show_pipeline = self.app.config.get('General', 'show_pipeline') == 'On'
        image = (
            self.active_pipeline.image
            if show_pipeline else self.active_pipeline.original_with_contours
        )

        match image.ndim:
            case 2:  # Grayscale
                colorfmt = 'luminance'
            case 3:  # BGR
                colorfmt = 'bgr'
            case _:  # Should not happen
                raise ValueError('Bad image shape')

        buff = cv.flip(image, 0).tobytes()
        texture = Texture.create(
            size=(image.shape[1], image.shape[0]), colorfmt=colorfmt
        )
        texture.blit_buffer(buff, colorfmt=colorfmt, bufferfmt='ubyte')

        self.image.texture = texture
        self.image.opacity = 1
        self.center_image()

    def sync_texture(self, interval):
        if self.active_pipeline is not None:
            self.update_image()

    def clear_image_and_pipeline(self):
        self.active_pipeline = None
        self.image.texture = None
        self.image.opacity = 0

    def resize_image(self, instance, value):
        if self.active_pipeline is not None:
            self.active_pipeline.resize(value)
    ##############################
    # Pipeline operations
    ##############################
    def gray_image(self, instance, value):
        if self.active_pipeline is not None:
            image = self.active_pipeline.image
            if image.ndim == 3:
                self.active_pipeline.gray()

    def blur_image(self, instance, value):
        if self.active_pipeline is not None:
            self.active_pipeline.blur(value)

    def detect_edges(self, instance, value):
        if self.active_pipeline is not None:
            self.active_pipeline.edges(value)

    def draw_contours(self, instance, value):
        if self.active_pipeline is not None:
            self.active_pipeline.draw_contours(value)

    def clear_contours(self, instance, value):
        if self.active_pipeline is not None:
            self.active_pipeline.clear_contours(value)

    ##############################
    # Image operations
    ##############################
    def center_image(self):
        if self.image.texture:
            self.image.center_x = 0.5 * self.width
            self.image.center_y = 0.5 * self.height

    def zoom(self, factor):
        self.scatter.scale *= factor
        # self.center_image()


class MPLApp(App):
    title = 'MPLCV'

    def build(self):
        self.mpl_widget = MPLWidget(app=self)
        return self.mpl_widget

    def build_config(self, config):
        config.adddefaultsection('General')
        config.setdefault('General', 'show_pipeline', 'Off')

        config.adddefaultsection('Visuals')

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


if __name__ == '__main__':
    MPLApp().run()
