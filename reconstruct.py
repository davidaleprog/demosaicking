"""The main file for the reconstruction.
"""

import numpy as np

from src.forward_model import CFA
from src.methods.david_alexis.functions import bayer_gradient_interpolation, quad_gradient_interpolation


def run_reconstruction(y: np.ndarray, cfa: str) -> np.ndarray:
    """Performs demosaicking on y.

    Args:
        y (np.ndarray): Mosaicked image to be reconstructed.
        cfa (str): Name of the CFA. Can be bayer or quad_bayer.

    Returns:
        np.ndarray: Demosaicked image.
    """
    input_shape = (y.shape[0], y.shape[1], 3)
    op = CFA(cfa, input_shape)

    if cfa == "bayer": 
        res = bayer_gradient_interpolation(y, op)
    else:
        res = quad_gradient_interpolation(y, op)

    return res



# Author: Alexis DAVID
