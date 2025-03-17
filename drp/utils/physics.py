
import numpy as np
import re
import os
import gc
import math
import time
import logging
from scipy import ndimage
from enum import Enum
from drp.utils.memmap import DataModality, load_cube
from drp.utils.generate_offset import generate_offset

# cst for dela_rho calculation
CS_SQUARE = 0.3


def visco_from_filename(name):
    nu_re = re.search(r"(nu_\d{1}p\d+)", name)
    return float(nu_re.groups()[0].split("_")[1].replace("p", "."))


def compute_poro_perm(root, offset=None, subshape=None, nu_override=0.01):
    """
    compute porosity and permeability from a data dir
    if offset is None, the whole volume is used
    :param root: data root dir
    :param offset: numpy shape like tuple
    :param subshape: numpy shape like tuple
    :return: computed porosity and permeability
    """

    # t0 = time.time()

    # print(f"Compute permeability for : {root} , offset : {offset} , subshape : {subshape}")
    # load arrays
    binary = load_cube(
        root, modality=DataModality.BIN, offset=offset, subshape=subshape
    )
    dens = load_cube(root, modality=DataModality.DENS, offset=offset, subshape=subshape)
    jvstps = load_cube(
        root, modality=DataModality.JVSTPS, offset=offset, subshape=subshape
    )

    # In the following :
    # * binary == 0 means [] => jvstps_mean = nan
    # * binary[0] == 0 means [] => moy1 = nan
    # * binary[-1] == 0 means [] => moy2 = nan
    # Small cube shapes could do this
    try:
        # compute permeability
        # porosity phi
        phi = np.sum(binary == 0) / binary.size
        # print(f"porosity : {phi}")
        # viscosity
        # nu = 0.01 # deprecated : was valid for the first dataset
        # parse viscosity from filename
        try:
            jvstps_filename = next(
                filter(
                    lambda path: str(DataModality.JVSTPS).lower() in path.lower()
                    and path.lower().endswith(".dat"),
                    os.listdir(root),
                ),
                None,
            )
            nu = visco_from_filename(jvstps_filename)
            print(f"Parsing viscosity :  nu = {nu}") # for debugging
        except Exception as err:
            print(f"Error while parsing viscosity from file : {err}")
            print(f"Setting viscosity to nu = {nu_override}")
            nu = nu_override

        # print(f"viscosity : {nu}")
        _jvstps = jvstps[binary == 0]
        if len(_jvstps) == 0:
            return -1, -1
        # site solide à 1 sur site image segmentée ( les moyennes sont sur les voxels à 0 )
        # flux moyen
        jvstps_mean = np.mean(_jvstps)
        # print(f"jvstps_mean : {jvstps_mean}")
        # densite
        left = dens[0][binary[0] == 0]
        if len(left) == 0:
            return -1, -1
        moy1 = np.mean(left)
        right = dens[-1][binary[-1] == 0]
        if len(right) == 0:
            return -1, -1
        moy2 = np.mean(right)
        # print(f'mean density input slice  {moy1}')
        # print(f'mean density input slice {moy2}')
        delta_rho = moy1 - moy2
        # print(f"delta_rho {delta_rho}")
        delta_P = delta_rho * CS_SQUARE
        # print(f"delta_P {delta_P}")
        # L is a the depth of the volume in voxels
        L = binary.shape[0]
        # print(f"L {L}")
        # t1 = time.time() - t0
        # print(f"Time elapsed permeability calculation: {t1}")
        return phi, phi * nu * jvstps_mean * L / delta_P
    except FloatingPointError:
        return -1, -1


def generate_offset_and_label(
    root, fullshape, subshape, bound_min=0, bound_max=math.inf
):
    def _generate():
        try:
            offset_ = generate_offset(fullshape, subshape)
            _, perm_ = compute_poro_perm(root=root, subshape=subshape, offset=offset_)
            return offset_, perm_
        except FloatingPointError:
            return [], -1

    offset, perm = _generate()
    while perm < bound_min or perm > bound_max:
        offset, perm = _generate()
    return offset, perm


if __name__ == "__main__":
    root = "C:/Users/nguyenva/Downloads/data_DRP/4419"
    heat_map = binary = load_cube(
        root, modality=DataModality.K_HEATMAP, offset=None, subshape=None
    )
    print(heat_map[1000, 1000, 2700])
    # offset = generate_offset(fullshape=[1100, 1100, 2800], subshape=[100, 100, 100])
    # phi, per = compute_poro_perm(root, subshape=(100, 100, 100), offset=(0, 200, 400))
    # print(per)