import os.path
import cv2 as cv

import kivy

from kivy.app import App

from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup

from kivy.properties import ObjectProperty

from kivy.graphics.texture import Texture

from kivy.clock import Clock

from kivy.logger import Logger, LOG_LEVELS

from pipeline import Pipeline

kivy.require('2.0.0')
Logger.setLevel(LOG_LEVELS["debug"])


class Background(Image):
    pass


class FileChooserContent(BoxLayout):
    file_chooser = ObjectProperty()
    load = ObjectProperty()
    cancel = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_chooser.path = kwargs.get('path', os.path.expanduser('~'))


class MPLWidget(Widget):
    app = ObjectProperty()
    layout = ObjectProperty()

    image = Image(opacity=0)

    pipelines = []  # Store active opencv states
    active_pipeline = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layout.add_widget(self.image)

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
                image = cv.imread(selection[0])
                if image is not None:
                    self.active_pipeline = Pipeline(image)

                    self.pipelines.append(self.active_pipeline)
                    self.update_image()

                    # Sync at 30 FPS
                    Clock.schedule_interval(self.sync_texture, 1 / 30)

                    self.dismiss_file_chooser()

    def dismiss_file_chooser(self):
        self.file_chooser.dismiss()

    def update_image(self):
        image = self.active_pipeline.image

        buff = cv.flip(image, 0).tostring()
        texture = Texture.create(
            size=(image.shape[1], image.shape[0]), colorfmt='bgr'
        )
        texture.blit_buffer(buff, colorfmt='bgr', bufferfmt='ubyte')

        self.image.texture = texture
        self.image.opacity = 1

    def sync_texture(self, interval):
        if self.active_pipeline is not None:
            self.update_image()

    def clear_image_and_pipeline(self):
        self.active_pipeline = None
        self.image.texture = None
        self.image.opacity = 0

    def on_bin_button_press(self):
        self.clear_image_and_pipeline()

class MPLApp(App):
    title = 'matplotcv'

    def build(self):
        self.mpl_widget = MPLWidget(app=self)
        return self.mpl_widget

    def build_config(self, config):
        config.adddefaultsection('General')

        config.adddefaultsection('Visuals')

    def build_settings(self, settings):
        settings.add_json_panel(
            'General', self.config, data='''
            [
                {"type": "title", "title": "General"}
            ]
            '''
        )


if __name__ == '__main__':
    MPLApp().run()