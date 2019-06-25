# -*- coding: utf-8 -*-
"""
Module acting as a wrapper for the C-compiled functions that make up the core
of the QuakeMigrate package.

"""

import os
import pathlib

import numpy as np
import numpy.ctypeslib as clib

c_int = clib.ctypes.c_int
c_int8 = clib.ctypes.c_int8
c_int16 = clib.ctypes.c_int16
c_int32 = clib.ctypes.c_int32
c_int64 = clib.ctypes.c_int64
c_dbl = clib.ctypes.c_double

c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_i8Pt = clib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
c_i16Pt = clib.ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")
c_i32Pt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_i64Pt = clib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
c_iPt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

p = pathlib.Path(__file__).parents[1] / "lib/QMigrate"
if os.name == 'nt':
    _qmigratelib = clib.load_library(str(p.with_suffix(".dll")), ".")
else:  # posix
    _qmigratelib = clib.load_library(str(p.with_suffix(".so")), ".")


_qmigratelib.scan4d.argtypes = [c_dPt, c_i32Pt, c_dPt, c_int32, c_int32,
                                c_int32, c_int32, c_int64, c_int64]


def migrate(sig, tt, fsmp, lsmp, nsamp, map4d, threads):
    """
    Wrapper for the C-compiled scan4d function: computes 4-D coalescence map
    by back-migrating P and S onset functions.

    Returns output by populating map4d.

    Parameters
    ----------
    sig : array-like
        P and S onset functions

    tt : array-like
        P and S travel-time lookup-tables

    fsmp : int
        First sample in array to scan from

    lsmp : int
        Last sample in array to scan upto

    nsamp : int
        Number of samples in array to scan over

    map4d : array-like
        Empty array with shape of 4-D coalescence map that will be output

    threads : int
        Number of threads to perform the scan on

    Raises
    ------
    ValueError
        If there is a mismatch between number of stations in sig and look-up
        table

    ValueError
        If the 4-D array is too small

    ValueError
        If the sig array is smaller than map4d[0, 0, 0, :]

    """

    nstn, ssmp = sig.shape

    if not tt.shape[-1] == nstn:
        msg = "Mismatch between number of stations for data and LUT, {} - {}"
        msg = msg.format(nstn, tt.shape[-1])
        raise ValueError(msg)

    ncell = tt.shape[:-1]
    tcell = np.prod(ncell)

    if map4d.size < nsamp*tcell:
        msg = "4D-array is too small."
        raise ValueError(msg)

    if sig.size < nsamp + fsmp:
        msg = "Data array smaller than coalescence array."
        raise ValueError(msg)

    _qmigratelib.scan4d(sig, tt, map4d, c_int32(fsmp), c_int32(lsmp),
                        c_int32(nsamp), c_int32(nstn),  c_int64(tcell),
                        c_int64(threads))


_qmigratelib.detect4d.argtypes = [c_dPt, c_dPt, c_i64Pt, c_int32,
                                  c_int32, c_int32, c_int64, c_int64]


def find_max_coa(map4d, max_coa, grid_index, fsmp, lsmp, threads):
    """
    Wrapper for the C-compiled detect4d function: finds the maximum
    coalescence value in the 4-D coalesence grid at each time-step.

    Returns output by populating max_coa and grid_index.

    Parameters
    ----------
    map4d : array-like
        4-D coalescence map

    max_coa : array-like, double
        empty array with shape of max coa values that will be output

    grid_index : 

    fsmp : int
        First sample in array to scan from

    lsmp : int
        Last sample in array to scan upto

    threads : int
        Number of threads to perform the scan on

    Raises
    ------
    ValueError
        If the output array size is too small

    """

    nsamp = map4d.shape[-1]
    ncell = np.prod(map4d.shape[:-1])

    if max_coa.size < nsamp or grid_index.size < nsamp:
        msg = "Output array size too small, sample count = {}."
        msg = msg.format(nsamp)
        raise ValueError(msg)

    _qmigratelib.detect4d(map4d, max_coa, grid_index, c_int32(fsmp),
                          c_int32(lsmp), c_int32(nsamp), c_int64(ncell),
                          c_int64(threads))
