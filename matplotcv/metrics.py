import math
import numpy as np
from scipy.linalg import solve


def point_segment_distance(
    point: tuple[float], segment: tuple[tuple[float]]
) -> float:
    '''Calculate the distance between a point and a line segment.'''
    x, y = point
    sx, sy = segment[0]
    ex, ey = segment[1]

    length = (ex - sx)**2 + (ey - sy)**2
    if length == 0:  # Segment's start and end points are the same
        return math.sqrt((x - sx)**2 + (y - sy)**2)

    # Calculate the projection of the point onto the line
    tau = max(
        0, min(1, ((x - sx) * (ex - sx) + (y - sy) * (ey - sy)) / length)
    )
    proj = (sx + tau * (ex - sx), sy + tau * (ey - sy))

    return math.sqrt((x - proj[0])**2 + (y - proj[1])**2)


def affine_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute the augmented matrix for affine transformation from x to y.

    The affine transformation from x to y is given by
        [y0, y1] = T @ [x0, x1, 1],
    where T is the augmented affine transformation matrix, i.e.
        T = [[a00, a01, b0]
             [a10, a11, b1]
             [0,     0,  1]].
    '''
    x = np.concatenate([x, np.ones([1, x.shape[1]])], axis=0)
    return solve(x.T, y.T).T