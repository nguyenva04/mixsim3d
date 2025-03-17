import numpy as np
import functools


def is_valid_offset(subshape, offset, fullshape):
    """
    Check if the required offset and subshape are within data shape
    :return:
    """
    return functools.reduce(
        lambda x, y: x and y,
        map(lambda x, y, z: 0 <= (x + y) <= z, subshape, offset, fullshape),
    )


def generate_offset(fullshape, subshape):
    random_offset = tuple(map(lambda n: np.random.randint(low=0, high=n), fullshape))
    while not is_valid_offset(subshape, random_offset, fullshape):
        random_offset = tuple(
            map(lambda n: np.random.randint(low=0, high=n), fullshape)
        )
    return random_offset