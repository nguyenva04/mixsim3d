import os
from enum import Enum
from ast import literal_eval

import numpy as np
import functools


class DataModality(Enum):
    BIN = 1
    GLV = 2
    DENS = 3
    JVSTPS = 4
    K_HEATMAP = 5
    PHI_HEATMAP = 6

    def __str__(self):
        if self.value == 1:
            return "bin"
        elif self.value == 2:
            return "16bit"
        elif self.value == 3:
            return "dens"
        elif self.value == 4:
            return "jVsTps"
        elif self.value == 5:
            return "k_heatmap"
        elif self.value == 6:
            return "phi_heatmap"


def load_cube_meta_info(root, modality):
    """
    Return perm data volume meta info
    :param root:
    :param modality:
    :return: numpy type and shape
    """
    roi_file = next(
        filter(
            lambda path: str(modality).lower() in path.lower()
            and path.lower().endswith(".dat"),
            os.listdir(root),
        ),
        None,
    )
    assert (
        roi_file is not None
    ), f"Memmap data not found for modality {modality} and root {root}"
    cube_params = roi_file.split("-")
    cube_shape = literal_eval(cube_params[2].split(".")[0])
    return np.dtype(cube_params[1]), cube_shape


def is_valid_offset(subshape, offset, fullshape):
    """
    Check if the required offset and subshape are within data shape
    :return:
    """
    # test = list(map(lambda x, y, z: 0 <= (x + y) < z, subshape, offset, cube_shape))
    return functools.reduce(
        lambda x, y: x and y,
        map(lambda x, y, z: 0 <= (x + y) <= z, subshape, offset, fullshape),
    )


def check_meta_info(root, volumes, modality=DataModality.GLV):
    metas = [
        load_cube_meta_info(root=os.path.join(root, v), modality=modality)
        for v in volumes
    ]

    # use set() to sort duplicates
    meta = set(metas)

    assert len(meta) == 1, f"Volumes meta data are not identical (metas : {metas})"


def load_cube(root, modality, offset=None, subshape=None):
    """
    :param root: permeability data directory
    :param modality:  perm data type  BIN = 1 ,GLV = 2 ,DENS = 3, JVSTPS = 4,
    :param offset: numpy shape format (depth_offset, height_offset, width_offset)
    :param subshape: numpy shape format of the required slice of data
    :return: perm data in a numpy array
    """
    roi_file = next(
        filter(
            lambda path: str(modality).lower() in path.lower()
            and path.lower().endswith(".dat"),
            os.listdir(root),
        ),
        None,
    )
    assert (
        roi_file is not None
    ), f"Memmap data not found for modality {modality} and root {root}"

    cube_params = roi_file.split("-")
    cube_shape = literal_eval(cube_params[2].split(".")[0])

    # load data as memmap array
    memmap_array = np.memmap(
        os.path.join(root, roi_file),
        dtype=np.dtype(cube_params[1]),
        mode="r",
        shape=cube_shape,
    )
    if offset is None:
        offset = (0, 0, 0)

    if subshape is None:
        subshape = cube_shape

    if is_valid_offset(subshape, offset, cube_shape):
        memmap_array = memmap_array[
            offset[0] : subshape[0] + offset[0],
            offset[1] : subshape[1] + offset[1],
            offset[2] : subshape[2] + offset[2],
        ]
    else:
        raise Exception(
            f"Subshape {subshape} and offset {offset} are not valid for shape {cube_shape}"
        )

    return memmap_array
