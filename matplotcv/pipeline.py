import os.path
import warnings
from dataclasses import dataclass, field
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


@dataclass
class Contour:
    points: np.ndarray
    label: str = field(default='', init=False)
    closed: bool = False
    roi: np.ndarray | None = field(default=None, init=False)
    children: set[int] = field(default_factory=set, init=False)


class Pipeline:
    '''Pipeline controls all OpenCV computations.'''
    _processed = None
    _original = None
    isedgy = False
    blurring = 0
    contours = {}

    @property
    def processed(self):
        return self._processed

    @property
    def original(self):
        return self._original

    @property
    def isempty(self):
        return self.original is None and self.processed is None

    @property
    def isgray(self):
        return self.processed.ndim == 2

    @property
    def aspect(self):
        if not self.isempty:
            return self.original.shape[1] / self.original.shape[0]

    def load_image(self, filename: str):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in supported_exts:
            raise ValueError(f'Unsupported file extension {ext}')

        image = cv.imread(filename)

        if image is not None:
            self._original = image
            self._processed = image.copy()

    def clear(self, which: str = 'all'):
        if not self.isempty:
            match which:
                case 'all':
                    self._processed = self._original = None
                case 'processed':
                    self._processed = self._original.copy()
                case _:
                    raise ValueError('Bad clear option')

            self.blurring = 0
            self.isedgy = False
            self.contours = {}

    def resize(self, size: str):
        self.clear('processed')

        h, w = self.processed.shape[0], self.processed.shape[1]
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

        self._original = cv.resize(self.processed, (target_width, target_height))
        self._processed = self._original.copy()

    def gray(self):
        if not self.isempty and not self.isgray:
            self._processed = cv.cvtColor(self.processed, cv.COLOR_BGR2GRAY)

    def blur(self, kind: str = 'gaussian', n: int = 1):
        match kind:
            case 'gaussian':
                k = 3 + 2 * n
                self._processed = cv.GaussianBlur(self._processed, (k, k), 0)
            case _:
                raise ValueError('Bad blur function')
        self.blurring += k

    def edges(self, kind: str = 'canny'):
        match kind:
            case 'canny':
                # Automatic thresholding based on median
                sigma = 0.33
                m = np.median(self._processed)
                lower = int(max(0, (1.0 - sigma) * m))
                upper = int(min(255, (1.0 + sigma) * m))

                self._processed = cv.Canny(self._processed, lower, upper)
            case _:
                raise ValueError('Bad edge detection function')

        self.isedgy = True

    def find_contours(self, external: bool = False, key: int | None = None):
        mode = cv.RETR_EXTERNAL if external else cv.RETR_TREE

        if key is None:  # Search in the entire image
            contours, _ = cv.findContours(
                self._processed, mode, cv.CHAIN_APPROX_SIMPLE
            )
            self.contours = {i: Contour(c) for i, c in enumerate(contours)}
        else:  # Search in the parent contour
            self.contour_roi(key)
            contours, _ = cv.findContours(
                self.contours[key].roi, mode, cv.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                idx = [  # Check if the contour already exists
                    i for i, c in self.contours.items()
                    if np.array_equal(contour, c)
                ]
                if idx:
                    self.contours[key].children.add(idx[0])
                else:
                    idx = max(self.contours) + 1
                    self.contours[idx] = Contour(contour)
                    self.contours[key].children.add(idx)


    def split_contour(self, key: int, epsilon: float = 5.0) -> list[int]:
        '''
        Split contour at corners to obtain subcontours.
        '''
        contour = self.contours.get(key)
        if contour is None:
            raise ValueError(f'Contour {key} not found')

        points = contour.points

        corners = cv.approxPolyDP(
            np.array(points, dtype=np.int32).reshape([-1, 1, 2]),
            epsilon,
            contour.closed,
        )

        contours, current = [], []
        for p in points:
            current.append(p)
            if any(np.array_equal(p, corner) for corner in corners):
                contours.append(
                    np.array(current, dtype=np.int32).reshape(-1, 1, 2)
                )
                current = [p]

        # Create unique keys for new contours
        keys = [max(self.contours) + i + 1 for i in range(len(contours))]
        self.contours.update({k: Contour(c) for k, c in zip(keys, contours)})
        self.contours.pop(key)
        return keys

    def contour_roi(self, key: int, padding: int = 10):
        if self.contours.get(key) is None:
            raise ValueError(f'Contour {key} not found')

        if self.contours[key].roi is None:
            x, y, w, h = cv.boundingRect(self.contours[key].points)
            x, y, w, h = (
                max(0, x - padding),
                max(0, y - padding),
                min(self.processed.shape[1], x + w + padding),
                min(self.processed.shape[0], y + h + padding),
            )

            self.contours[key].roi = self.processed[y : h, x : w]