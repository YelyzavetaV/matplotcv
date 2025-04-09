import os.path
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
from exceptions import PipelineError, pipeline_error_handler
from utils import standard_coordinate

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
    '''Stores contour points and additional information.'''
    points: np.ndarray
    label: str = field(default='', init=False)
    _coordinate: tuple | None = field(default=None, init=False)
    closed: bool = False
    roi: tuple | None = field(default=None, init=False)
    children: set[int] = field(default_factory=set, init=False)

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, value: str):
        try:
            x, y = value.split(',')
            self._coordinate = (standard_coordinate(x), standard_coordinate(y))
        except Exception as e:
            raise ValueError(f'Invalid coordinate format: {value}') from e

        print(f'Coordinate set to {self.coordinate}')


class Pipeline:
    '''Pipeline controls all OpenCV computations.'''

    def __init__(self, image: np.ndarray | None = None):
        self.processed = self.original = image
        self.isedgy = False
        self.blurring = 0
        self.contours = {}

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

    def kernel_size(self, s: float = 0.01):
        if not self.isempty:
            return max(
                3,
                int(s * min(self.processed.shape[:2])) // 2 * 2 + 1
            )

    def load_image(self, filename: str):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in supported_exts:
            raise ValueError(f'Unsupported file extension {ext}')

        self.original = cv.imread(filename)
        self.processed = self.original.copy()

        if self.isempty:
            raise PipelineError('Could not load image')

    @pipeline_error_handler
    def process(self, filename: str | None = None):
        if filename is not None:
            self.clear()
            self.load_image(filename)

        self.resize('hd')
        self.gray()
        self.blur(strength=0.01)
        self.edges()

    def clear(self, which: str = 'all'):
        if not self.isempty:
            match which:
                case 'all':
                    self.processed = self.original = None
                case 'processed':
                    self.processed = self.original.copy()
                case _:
                    raise ValueError('Bad clear option')

            self.blurring = 0
            self.isedgy = False
            self.contours = {}

    def resize(self, size: str):
        if not self.isempty:
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

            self.original = cv.resize(
                self.processed, (target_width, target_height)
            )
            self.processed = self.original.copy()

    def gray(self):
        if not self.isempty and not self.isgray:
            self.processed = cv.cvtColor(self.processed, cv.COLOR_BGR2GRAY)

    def blur(
        self,
        kind: str = 'gaussian',
        n: int | None = None,
        strength: float = 0.01,
    ):
        if not self.isempty:
            if (
                not isinstance(n, int)
                and not isinstance(strength, (int, float))
            ):
                raise ValueError('Either n or strength must be provided')

            match kind:
                case 'gaussian':
                    if n is not None:
                        kernel_size = 3 + 2 * (n - 1)
                    else:
                        kernel_size = self.kernel_size(strength)

                    self.processed = cv.GaussianBlur(
                        self.processed, (kernel_size, kernel_size), 0
                    )
                case _:
                    raise ValueError('Bad blur function')
            self.blurring += kernel_size

    def edges(self, kind: str = 'canny'):
        if not self.isempty:
            match kind:
                case 'canny':
                    # Automatic thresholding based on median
                    sigma = 0.33
                    m = np.median(self.processed)
                    lower = int(max(0, (1.0 - sigma) * m))
                    upper = int(min(255, (1.0 + sigma) * m))

                    self.processed = cv.Canny(self.processed, lower, upper)
                case _:
                    raise ValueError('Bad edge detection function')

            self.isedgy = True

    def find_contours(self, external: bool = False, key: int | None = None):
        if not self.isempty:
            if key is None:  # Search in the entire image
                contours, _ = cv.findContours(
                    self.processed,
                    cv.RETR_EXTERNAL if external else cv.RETR_TREE,
                    cv.CHAIN_APPROX_SIMPLE,
                )
                self.contours = OrderedDict(
                    (i, Contour(c)) for i, c in enumerate(contours)
                )
            else:  # Search in the parent contour
                self.contour_roi(key)
                x, y, w, h = self.contours[key].roi
                roi = self.processed[y:h, x:w]

                contours, _ = cv.findContours(
                    roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    # Translate contour to the original image coordinates
                    contour += np.array([x, y])

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

    def contour_roi(self, key: int, fraction: float = 0.05):
        if self.contours.get(key) is None:
            raise ValueError(f'Contour {key} not found')

        if self.contours[key].roi is None:
            px, py = (
                int(fraction * self.processed.shape[1]),
                int(fraction * self.processed.shape[0]),
            )

            x, y, w, h = cv.boundingRect(self.contours[key].points)
            x, y, w, h = (
                max(0, x - px),
                max(0, y - py),
                min(self.processed.shape[1], x + w + px),
                min(self.processed.shape[0], y + h + py),
            )
            self.contours[key].roi = (x, y, w, h)
