import os.path
import warnings
import numpy as np
import cv2 as cv

supported_exts = (
    '.png',
    '.jpg',
    '.jpeg',
    '.tiff',
    '.tif',
    '.bmp',
    '.ppm',
    '.pgm',
    '.pbm',
    '.webp',
)

sizes = {
    'vga': (640, 480),
    'hd': (1280, 720),
    'fhd': (1920, 1080),
    '4k': (3840, 2160),
}


class Pipeline:
    '''Pipeline controls all OpenCV computations.'''
    _image = None
    _original = None
    edges_detected = False
    contours = []

    @property
    def empty(self):
        return self.original is None and self.image is None

    @property
    def image(self):
        return self._image

    @property
    def original(self):
        return self._original

    @property
    def aspect(self):
        if not self.empty:
            return self.original.shape[1] / self.original.shape[0]

    def load_image(self, filename: str):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in supported_exts:
            raise ValueError(f'Unsupported file extension {ext}')

        image = cv.imread(filename)

        if image is not None:
            self._original = image
            self._image = image.copy()

    def clear(self, which: str = 'all'):
        if not self.empty:
            match which:
                case 'all':
                    self._image = self._original = None
                case 'processed':
                    self._image = self._original.copy()
                case _:
                    raise ValueError('Bad clear option')

            self.edges_detected = False
            self.contours = []

    def resize(self, size: str):
        self.clear('processed')

        h, w = self.image.shape[0], self.image.shape[1]
        aspect_ratio = w / h

        try:
            target_width, target_height = sizes[size]
        except KeyError as e:
            raise ValueError(f'Size "{size}" not supported') from e

        if w >= h:
            target_height = int(target_width / aspect_ratio)
        else:
            target_width = int(target_height * aspect_ratio)

        if target_width > w:
            warnings.warn('Cannot increase the size of the image')
            return

        self._original = cv.resize(self.image, (target_width, target_height))
        self._image = self._original.copy()

    def gray(self):
        if not self.empty:
            if self.image.ndim == 3:
                self._image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def blur(self, kind: str = 'gaussian', n: int = 1):
        match kind:
            case 'gaussian':
                k = 3 + 2 * n
                self._image = cv.GaussianBlur(self._image, (k, k), 0)
            case _:
                raise ValueError('Bad blur function')

    def edges(self, kind: str = 'canny'):
        match kind:
            case 'canny':
                # Automatic thresholding based on median
                sigma = 0.33
                m = np.median(self._image)
                lower = int(max(0, (1.0 - sigma) * m))
                upper = int(min(255, (1.0 + sigma) * m))

                self._image = cv.Canny(self._image, lower, upper)
            case _:
                raise ValueError('Bad edge detection function')

        self.edges_detected = True

    def contour_tree(self, which: str = 'all'):
        match which:
            case 'all':
                mode = cv.RETR_TREE
            case 'external':
                mode = cv.RETR_EXTERNAL
            case _:
                raise ValueError('Bad contour mode')

        self.contours, _ = cv.findContours(
            self._image, mode, cv.CHAIN_APPROX_SIMPLE
        )
