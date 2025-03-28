import math


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
