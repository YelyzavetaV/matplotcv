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
    image = None

    def __init__(self, image: np.ndarray):
        self._original = image
        self.original_with_contours = image.copy()
        self.image = image.copy()

        self.h, self.w = self.image.shape[0], self.image.shape[1]
        self.c = self.image.shape[2] if self.image.ndim == 3 else 1
        self.edges_detected = False

        self.k = 0  # no blur

        self.contours = []
        self.hierarchy = []

    @classmethod
    def from_file(cls, filename: str):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in supported_exts:
            raise ValueError(f'Unsupported file extension {ext}')

        image = cv.imread(filename)

        if image is not None:
            return cls(image)

    @property
    def original(self):
        return self._original

    def resize(self, size: str):
        aspect_ratio = self.w / self.h

        try:
            target_width, target_height = sizes[size]
        except KeyError as e:
            raise ValueError(f'Size "{size}" not supported') from e

        if self.w >= self.h:
            target_height = int(target_width / aspect_ratio)
        else:
            target_width = int(target_height * aspect_ratio)

        if target_width > self.w:
            warnings.warn('Cannot increase the size of the image')
            return

        self.image = cv.resize(self.image, (target_width, target_height))
        self.h, self.w = self.image.shape[0], self.image.shape[1]

    def gray(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def blur(self, kind: str = 'gaussian', n: int = 1):
        match kind:
            case 'gaussian':
                k = 3 + 2 * n
                self.image = cv.GaussianBlur(self.image, (k, k), 0)
            case _:
                raise ValueError('Bad blur function')

        self.k += k

    def edges(self, kind: str = 'canny'):
        match kind:
            case 'canny':
                # Automatic thresholding based on median
                sigma = 0.33
                m = np.median(self.image)
                lower = int(max(0, (1.0 - sigma) * m))
                upper = int(min(255, (1.0 + sigma) * m))

                self.image = cv.Canny(self.image, lower, upper)
            case _:
                raise ValueError('Bad edge detection function')

        self.edges_detected = True

    def contour_tree(self):
        self.contours, self.hierarchy = cv.findContours(
            self.image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

    def draw_contours(self, which: str = 'all'):
        if not self.edges_detected:
            self.edges()

        if not self.contours:
            self.contour_tree()

        match which:
            case 'all':
                idx = -1
            case _:
                raise ValueError('Bad contour index')

        cv.drawContours(
            self.original_with_contours, self.contours, idx, (0, 255, 0), 3
        )

    def clear_contours(self, which: str = 'all'):
        self.original_with_contours = self.original.copy()

        match which:
            case 'all':
                self.contours = []
                self.hierarchy = []
            case _:  # TODO: Redraw contours
                raise ValueError('Bad contour index')
