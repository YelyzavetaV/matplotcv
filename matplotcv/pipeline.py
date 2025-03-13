import numpy as np
import cv2 as cv


class Pipeline:
    image = None

    def __init__(self, image: np.ndarray):
        self.image = image
        self.h, self.w = self.image[0], self.image[1]
        self.c = self.image[2] if self.image.ndim == 3 else 1
