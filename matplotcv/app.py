import os
import csv
import cv2 as cv
import numpy as np

import kivy
from kivy.factory import Factory
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger, LOG_LEVELS

from exceptions import PipelineError
from components import (
    ErrorPopup,
    ConfirmationPopup,
    FileLoadPopup,
    FileSavePopup,
    ToolsDropDown,
    MathDropDown,
    ContourWidget,
)
from pipeline import Pipeline
from metrics import affine_map

kivy.require('2.3.0')
Logger.setLevel(LOG_LEVELS['debug'])


class MPLWidget(Widget):
    app = ObjectProperty()
    layout = ObjectProperty()
    scatter = ObjectProperty()
    image = ObjectProperty()
    original_image_toggle = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pipeline = Pipeline()
        self.contours = {}
        self.marked_contours = set()
        self.drawn_contours = None
        self.transform_matrix = None

        # Initialize and bind components
        Window.bind(
            on_resize=self.on_window_resize, mouse_pos=self.on_mouse_move
        )

        self.file_loader = FileLoadPopup()
        self.file_loader.load = self.on_load_from_file_button_press

        self.file_saver = FileSavePopup()
        self.file_saver.save = self.on_save_to_file_button_press

        self.resize_dropdown = Factory.ResizeDropDown()
        self.resize_dropdown.bind(
            on_select=lambda i, v: self.pipeline.resize(v)
        )

        self.tools_dropdown = ToolsDropDown()
        self.tools_dropdown.bind(
            on_select=lambda i, v: self.pipeline.process()
        )
        self.tools_dropdown.bind(on_select=lambda i, v: self.pipeline.gray())
        self.tools_dropdown.blur_dropdown.bind(
            on_select=lambda i, v: self.pipeline.blur(v)
        )
        self.tools_dropdown.detect_edges_dropdown.bind(
            on_select=lambda i, v: self.pipeline.edges(v)
        )

        self.draw_dropdown = Factory.DrawDropDown()
        self.draw_dropdown.bind(
            on_select=lambda i, v: self.draw_contours(color=v)
        )

        self.math_dropdown = MathDropDown()

    #---------------------------
    # UI operations
    #---------------------------
    def on_window_resize(self, instance, w, h):
        '''Redraw all contours that are currently displayed.'''
        if self.contours:
            Clock.unschedule(self.draw_contours)  # Prevent multiple calls
            Clock.schedule_once(
                lambda dt: self.draw_contours(redraw=True), 0.1
            )

    def on_mouse_move(self, window, pos):
        '''Highlight a contour when the mouse is over it.'''
        threshold = self.app.config.get(
            'Advanced', 'contour_collide_threshold'
        )
        threshold = float(threshold)

        pos = self.image.to_widget(*pos)
        for contour in self.contours.values():
            contour.hovered = contour.collide_point(*pos, threshold)

    def on_load_from_file_button_press(self, selection):
        if selection:  # Try loading file once confirmed with load button
            if self.file_loader.load_button.state == 'down':
                self.clear()

                try:
                    self.pipeline.load_image(selection[0])
                except PipelineError:
                    error_popup = ErrorPopup()
                    error_popup.message = 'Could not load image'
                    error_popup.open()

                self.file_loader.dismiss()

                self.update_image()

                # Sync at 30 FPS
                Clock.schedule_interval(
                    lambda interval: self.update_image(), 1 / 30
                )

    def on_save_to_file_button_press(self, dir: str, name: str):
        path = os.path.join(dir, name)

        if not path.endswith('.csv'):
            path += '.csv'

        if os.path.exists(path):
            confirmation_popup = ConfirmationPopup()

            confirmation_popup.message = (
                'File already exists. Do you want to overwrite it?'
            )
            confirmation_popup.confirm = lambda: self.write_contour(path)

            confirmation_popup.open()
        else:
            self.write_contour(path)

    def on_original_image_toggle_press(self):
        if self.original_image_toggle.state == 'down':
            self.app.config.set('General', 'show_pipeline', 'ON')
        else:
            self.app.config.set('General', 'show_pipeline', 'OFF')
        self.app.config.write()

    def on_log_scale_select(self, value):
        self.app.config.set('Math', 'log_scale', value)
        self.app.config.write()

    #---------------------------
    # Image operations
    #---------------------------
    def resize_image(self):
        '''Dynamically resize the image to fit 0.75 of the window.'''
        w, h = self.size
        aspect = self.pipeline.aspect

        if w / h > aspect:
            self.image.size = (0.75 * h * aspect, 0.75 * h)
        else:
            self.image.size = (0.75 * w, 0.75 * w / aspect)

    def update_image(self):
        if not self.pipeline.isempty:
            show_pipeline = self.app.config.get(
                'General', 'show_pipeline'
            ) == 'ON'
            image = (
                self.pipeline.processed
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
            self.image.canvas.ask_update()

    def center_image(self):
        if self.image.texture:
            self.image.pos = (
                0.5 * (self.width - self.image.width),
                0.5 * (self.height - self.image.height),
            )

    def zoom(self, factor):
        self.scatter.scale *= factor

    def update_transform_matrix(self):
        ticks = []
        for j, contour in self.pipeline.contours.items():
            if not contour.label:
                break
            if contour.label == 'tick':
                ticks.append(j)

        if len(ticks) < 3:
            error_popup = ErrorPopup()
            error_popup.message = (
                'At least 3 ticks are required to construct '
                'transform matrix'
            )
            error_popup.open()
            return

        ek = np.array(
            [self.pipeline.contours[j].points[1][0] for j in ticks]
        ).T

        ep = np.array(
            [self.pipeline.contours[tick].coordinate for tick in ticks]
        ).T

        # Transformation matrix maps the image coordinates to the user
        # coordinates
        self.transform_matrix = affine_map(ek, ep)

        return self.transform_matrix

    def clear(self):
        self.pipeline.clear('all')
        self.clear_contour(self.contours.keys())
        self.image.texture = None
        self.transform_matrix = None

    #---------------------------
    # Contour operations
    #---------------------------
    def map_cv_contour_to_image(self, contour: int | ContourWidget):
        '''Map OpenCV contour to widget coordinates.'''
        w, h = self.image.size
        x, y = self.image.pos

        scale = (
            w / self.pipeline.original.shape[1],
            h / self.pipeline.original.shape[0],
        )

        if isinstance(contour, int):
            contour = self.pipeline.contours.get(contour)

        return [
            (x + p[0][0] * scale[0], y + h - p[0][1] * scale[1])
            for p in contour.points
        ]

    def map_image_contour_to_user(self, key: int):
        self.update_transform_matrix()

        x = np.squeeze(self.pipeline.contours[key].points).T
        x = np.concatenate([x, np.ones([1, x.shape[1]])], axis=0)

        return self.transform_matrix @ x

    def draw_contours(
        self,
        color: str = 'blue',
        redraw: bool = False,
        contours: set[int] | None = None
    ):
        '''Draw OpenCV contours as widgets'''
        p = self.pipeline

        if not p.isempty:
            if not p.isedgy:
                p.edges()
            if not p.contours:
                p.find_contours()

            contours = contours if contours is not None else p.contours.keys()

            if redraw:
                self.clear_contour(contours)

            for k in p.contours:
                if k not in self.contours:
                    contour = ContourWidget(
                        k, self.map_cv_contour_to_image(p.contours[k]), color
                    )
                    self.image.add_widget(contour)
                    self.contours[k] = contour

    def replace_contour(self, old: int, new: dict):
        if old in self.contours:
            self.clear_contour(old)

            for k in new:
                contour = ContourWidget(k, self.map_cv_contour_to_image(k))
                self.image.add_widget(contour)
                self.contours[k] = contour

    def split_contour(self, key: int):
        contour = self.pipeline.contours.get(key)
        if contour is None:  # Shouldn't happen
            raise RuntimeError('Contour not found')

        subkeys = self.pipeline.split_contour(key)

        self.replace_contour(
            key, {k: self.pipeline.contours[k]
                  for k in subkeys}
        )

    def label_contour(
        self,
        key: int,
        label: str | None = None,
        coordinate: str | None = None,
    ):
        if key not in self.pipeline.contours:  # Shouldn't happen
            raise RuntimeError('Contour not found')

        Logger.debug(f'Labeling contour {key} as {label} at {coordinate}')

        if label is not None:
            self.pipeline.contours[key].label = label
        if coordinate is not None:
            self.pipeline.contours[key].coordinate = coordinate

        # Labeled contours are stored at the beginning to then quickly
        # locate them
        self.pipeline.contours.move_to_end(key, last=False)

        self.draw_contours(color='red', redraw=True, contours={key})

    def write_contour(self, filename: str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])

            for key in self.marked_contours:
                Logger.debug(f'Exporting contour {key}')

                points = self.map_image_contour_to_user(key)
                writer.writerows(points.T)

        self.marked_contours.clear()

    def clear_contour(self, keys: int | set[int]):
        if isinstance(keys, int):
            keys = [keys]

        for key in list(keys):
            self.image.remove_widget(self.contours[key])
            self.contours.pop(key)
            self.marked_contours.discard(key)


class MPLApp(App):
    title = 'MPLCV'

    def build(self):
        Window.minimum_width = self.config.getint('Graphics', 'min_width')
        Window.minimum_height = self.config.getint('Graphics', 'min_height')

        self.root_widget = MPLWidget(app=self)
        return self.root_widget

    def build_config(self, config):
        config.adddefaultsection('Math')
        config.setdefault('Math', 'log_scale', 'OFF')

        config.adddefaultsection('General')
        config.setdefault('General', 'show_pipeline', 'OFF')

        config.adddefaultsection('Graphics')
        config.setdefault('Graphics', 'min_width', 800)
        config.setdefault('Graphics', 'min_height', 600)

        config.adddefaultsection('Advanced')
        config.setdefault('Advanced', 'contour_collide_threshold', 10)

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
                'values': ['ON', 'OFF']
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

        settings.add_json_panel(
            'Advanced',
            self.config,
            data='''
            [
                {'type': 'title', 'title': 'Advanced'},
            ]
            '''
        )

        settings.add_json_panel(
            'Math',
            self.config,
            data='''
            [
                {'type': 'title', 'title': 'Math'},
                {
                'type': 'options',
                'title': 'Logarithmic scale',
                'desc': 'Turn on/off the logarithmic scale',
                'section': 'Math',
                'key': 'log_scale',
                'options': ['OFF', 'X', 'Y', 'XY'],
                },
            ]
            '''
        )


if __name__ == '__main__':
    MPLApp().run()
