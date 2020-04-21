# -*- coding: utf-8 -*-
"""
Bindings for the core compiled C routines, migrate and find_max_coa.

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

p = pathlib.Path(__file__).parent / "src" / "QMigrate"
if os.name == "nt":
    _qmigratelib = clib.load_library(str(p.with_suffix(".dll")), ".")
else:  # posix
    _qmigratelib = clib.load_library(str(p.with_suffix(".so")), ".")


_qmigratelib.migrate.argtypes = [c_dPt, c_i32Pt, c_dPt, c_int32, c_int32,
                                 c_int32, c_int32, c_int32, c_int64, c_int64]


def migrate(sig, tt, fsmp, lsmp, nsamp, avail, map4d, threads):
    """
    Wrapper for the C-compiled migrate function: computes 4-D coalescence map
    by back-migrating P and S onset functions.

    Returns output by populating map4d.

    Parameters
    ----------
    sig : array-like
        P and S onset functions.
    tt : array-like
        P and S travel-time lookup tables.
    fsmp : int
        Sample in array from which to scan.
    lsmp : int
        Sample in array up to which to scan.
    nsamp : int
        Number of samples in array over which to scan.
    avail : int
        Number of available onset functions.
    map4d : array-like
        Empty array with shape of 4-D coalescence map that will be output.
    threads : int
        Number of threads with which to perform the scan.

    Raises
    ------
    ValueError
        If mismatch between number of stations in sig and lookup table.
    ValueError
        If the 4-D array is too small.
    ValueError
        If the sig array is smaller than map4d[0, 0, 0, :].

    """

    nstn, _ = sig.shape
    if not tt.shape[-1] == nstn:
        raise ValueError(("Mismatch between number of stations for data and "
                          "LUT, {} - {}".format(nstn, tt.shape[-1])))

    ncell = tt.shape[:-1]
    tcell = np.prod(ncell)

    if map4d.size < nsamp*tcell:
        raise ValueError("4-D array is too small.")

    if sig.size < nsamp + fsmp:
        raise ValueError("Data array smaller than coalescence array.")

    _qmigratelib.migrate(sig, tt, map4d, c_int32(fsmp), c_int32(lsmp),
                         c_int32(nsamp), c_int32(nstn), c_int32(avail),
                         c_int64(tcell), c_int64(threads))


_qmigratelib.find_max_coa.argtypes = [c_dPt, c_dPt, c_dPt, c_i64Pt, c_int32,
                                      c_int32, c_int32, c_int64, c_int64]


def find_max_coa(map4d, max_coa, n_max_coa, grid_index, fsmp, lsmp, threads):
    """
    Wrapper for the C-compiled find_max_coa function: finds the continuous
    maximum coalescence amplitude in the 4-D coalescence grid.

    Returns output by populating max_coa and grid_index.

    Parameters
    ----------
    map4d : array-like
        4-D coalescence map.
    max_coa : array-like, double
        Empty array with shape (nsamp,) in which the maximum coalescence values
        will be recorded.
    n_max_coa : array-like, double
        Empty array with shape (nsamp,) in which the normalised maximum
        coalescence values will be recorded.
    grid_index : array-like, int
        Empty array with shape (nsamp,) that will record the flattened index of
        the maximum coalescence value.
    fsmp : int
        Sample in array from which to scan.
    lsmp : int
        Sample in array up to which to scan.
    threads : int
        Number of threads with which to perform the scan.

    Raises
    ------
    ValueError
        If the output array size is too small.

    """

    nsamp = map4d.shape[-1]
    ncell = np.prod(map4d.shape[:-1])

    if max_coa.size < nsamp or grid_index.size < nsamp:
        raise ValueError(("Output array size too small, sample count = "
                          f"{nsamp}."))

    _qmigratelib.find_max_coa(map4d, max_coa, n_max_coa, grid_index,
                              c_int32(fsmp), c_int32(lsmp), c_int32(nsamp),
                              c_int64(ncell), c_int64(threads))
