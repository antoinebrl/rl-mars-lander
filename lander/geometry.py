import numpy as np


def l2dist(x, y):
    """
    Euclidean distance.
    """
    return np.linalg.norm(x - y, ord=2)


def ccw(x, y, z):
    """
    Counterclockwise angle formed by [x, y] and [x, z]
    Taken from https://stackoverflow.com/a/9997374
    """
    return (z[1] - x[1]) * (y[0] - x[0]) > (y[1] - x[1]) * (z[0] - x[0])


def segment_intersect(A, B, C, D):
    """
    Check if two segments, [A, B] and [C, D], intersect each other.
    Taken from # https://stackoverflow.com/a/9997374
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def is_inside_polygon(polygon, x: int, y: int, y2: int = 0):
    """
    Check if a point is inside the area of a polygon. The is inspired by surface
    https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
    but with a line shooting vertically.
    """
    pos = [x, y]
    top = [x, y2]
    nb_intersection = 0
    for start, end in zip(polygon, polygon[1:]):
        if segment_intersect(start, end, pos, top):
            nb_intersection += 1
    return nb_intersection % 2


def find_flat_segment(polygon):
    """
    Find beginning and end indexes of a horizontal section in the polygon.
    This section can be made of several consecutive segments.
    If the polygon contains several flat sections, this function only identify the first one.
    The polygon is a list of 2D coordinates (x, y)
    """
    start = -1
    end = -1
    for i, (a, b) in enumerate(zip(polygon[:-1], polygon[1:])):
        if a[1] == b[1]:
            if start < 0:
                start = i
            end = i + 1
    return [start, end]


def convert_to_fixed_length_polygon(polygon, n: int):
    """
    Represent a polygon with a certain number of nodes by breaking down the longest segments.
    The polygon is a list or nd.array of 2D coordinates (x, y)
    """
    if len(polygon) > n:
        raise ValueError("The polygon has more than {n} nodes")
    while len(polygon) < n:
        index_longest = np.argmax([l2dist(x, y) for x, y in zip(polygon, polygon[1:])])
        intermediate_point = [
            (polygon[index_longest][0] + polygon[index_longest + 1][0]) / 2,
            (polygon[index_longest][1] + polygon[index_longest + 1][1]) / 2,
        ]
        if isinstance(polygon, list):
            polygon.insert(index_longest + 1, intermediate_point)
        else:
            polygon = np.insert(polygon, index_longest + 1, intermediate_point, 0)
    return polygon
