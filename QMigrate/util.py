# -*- coding: utf-8 -*-
"""
Module to contain various utility functions and classes

"""

import time

import numpy as np


def gaussian_1d(x, a, b, c):
    """
    Create a 1-dimensional Gaussian function

    Parameters
    ----------
    x : array-like
        Array of x values

    a : float / int
        Amplitude (height of Gaussian)

    b : float / int
        Mean (centre of Gaussian)

    c : float / int
        Sigma (width of Gaussian)

    Returns
    -------
    f : function
        1-dimensional Gaussian function

    """

    f = a * np.exp(-1. * ((x - b) ** 2) / (2 * (c ** 2)))
    return f


def gaussian_3d(nx, ny, nz, sgm, K):
    """
    Create a 3-dimensional Gaussian function

    Parameters
    ----------
    nx : array-like
        Array of x values

    ny : array-like
        Array of y values

    nz : array-like
        Array of z values

    sgm : float / int
        Sigma (width of gaussian in all directions)

    Returns
    -------
    f : function
        3-dimensional Gaussian function

    """

    nx2 = (nx - 1) / 2
    ny2 = (ny - 1) / 2
    nz2 = (nz - 1) / 2
    x = np.linspace(-nx2, nx2, nx)
    y = np.linspace(-ny2, ny2, ny)
    z = np.linspace(-nz2, nz2, nz)
    ix, iy, iz = np.meshgrid(x, y, z, indexing="ij")

    if np.isscalar(sgm):
        sgm = np.repeat(sgm, 3)
    sx, sy, sz = sgm

    f = np.exp(- (ix * ix) / (2 * sx * sx)
               - (iy * iy) / (2 * sy * sy)
               - (iz * iz) / (2 * sz * sz)) + K
    return f


class Stopwatch(object):
    """
    Simple stopwatch to measure elapsed wall clock time.

    """

    def __init__(self):
        """Object initialisation"""
        self.start = time.time()

    def __call__(self):
        """Return time elapsed since object initialised"""
        msg = "    \tElapsed time: {:6f} seconds.\n".format(time.time()
                                                            - self.start)
        return msg


class ArchiveEmptyException(Exception):
    """Custom exception to handle empty archive"""

    def __init__(self):
        msg = "ArchiveEmptyException: No data was available for this timestep."
        super().__init__(msg)


class DataGapException(Exception):
    """
    Custom exception to handle case when all data has gaps
    for a given timestep

    """

    def __init__(self):
        msg = "DataGapException: All available data had gaps for this timestep."
        super().__init__(msg)
