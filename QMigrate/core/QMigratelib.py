# -*- coding: utf-8 -*-
"""
This module requires that 'numpy' is installed in your Python environment.

This module acts as a wrapper for the C-compiled functions that make up the
core of the QuakeMigrate package.

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


def scan(sig, tt, fsmp, lsmp, nsamp, map4d, threads):
    """
    Wrapper for the C-compiled scan4d function

    Parameters
    ----------
    sig : 

    tt : 

    fsmp : int
        First sample in array to scan from
    lsmp : int
        Last sample in array to scan to
    nsamp : int
        Number of samples in array to scan over
    map4d : 

    threads : int
        Number of threads to perform the scan on
    Raises
    ------
    ValueError
        If there is a mismatch between number of stations for data and look-up
        table
    ValueError
        If the 4-D array is too small
    ValueError
        If the data array is smaller than the coalescence array

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


def detect(mmap, dsnr, dind, fsmp, lsmp, threads):
    """
    Wrapper for the C-compiled detect4d function

    Parameters
    ----------
    mmap : 

    dsnr : 

    dind : 

    fsmp : 

    lsmp : 

    threads :


    Raises
    ------
    ValueError
        If the output array size is too small

    """

    nsamp = mmap.shape[-1]
    ncell = np.prod(mmap.shape[:-1])
    if dsnr.size < nsamp or dind.size < nsamp:
        msg = "Output array size too small, sample count = {}."
        msg = msg.format(nsamp)
        raise ValueError(msg)
    _qmigratelib.detect4d(mmap, dsnr, dind, c_int32(fsmp), c_int32(lsmp),
                          c_int32(nsamp), c_int64(ncell), c_int64(threads))
