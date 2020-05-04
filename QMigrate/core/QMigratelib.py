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


def migrate(onsets, ttimes, fsmp, lsmp, avail, threads):
    """
    Wrapper for the C-compiled migrate function: computes 4-D coalescence map
    by back-migrating P and S onset functions.

    Parameters
    ----------
    onsets : `numpy.ndarry`, shape(nstations, tsamps)
        P and S onset functions.
    ttimes : `numpy.ndarry`, shape(nx, ny, nz, nstations)
        P and S traveltime lookup tables.
    fsmp : int
        Index of first sample in array from which to scan.
    lsmp : int
        Index of last sample in array up to which to scan.
    avail : int
        Number of available onset functions.
    threads : int
        Number of threads with which to perform the scan.

    Returns
    -------
    map4d : `numpy.ndarray`, shape(nx, ny, nz, nsamp)
        4-D coalescence map.

    Raises
    ------
    ValueError
        If mismatch between number of stations in sig and lookup table.
    ValueError
        If the 4-D array is too small.
    ValueError
        If the sig array is smaller than map4d[0, 0, 0, :].

    """

    ncell = ttimes.shape[:-1]
    nstations, tsamp = onsets.shape
    if not ttimes.shape[-1] == nstations:
        raise ValueError("Mismatch between number of stations for data and "
                         f"LUT, {nstations} - {ttimes.shape[-1]}")
    nsamp = tsamp - fsmp - lsmp
    map4d = np.zeros(ncell + (nsamp,), dtype=np.float64)

    tcell = np.prod(ncell)
    if map4d.size < nsamp*tcell:
        raise ValueError("4-D array is too small.")

    if onsets.size < nsamp + fsmp:
        raise ValueError("Data array smaller than coalescence array.")

    _qmigratelib.migrate(onsets, ttimes, map4d, c_int32(fsmp), c_int32(lsmp),
                         c_int32(nsamp), c_int32(nstations), c_int32(avail),
                         c_int64(tcell), c_int64(threads))

    return map4d


_qmigratelib.find_max_coa.argtypes = [c_dPt, c_dPt, c_dPt, c_i64Pt, c_int32,
                                      c_int64, c_int64]


def find_max_coa(map4d, threads):
    """
    Wrapper for the C-compiled find_max_coa function: finds the continuous
    maximum coalescence amplitude in the 4-D coalescence grid.

    Parameters
    ----------
    map4d : `numpy.ndarray`, shape(nx, ny, nz, nsamples)
        4-D coalescence map.
    threads : int
        Number of threads with which to perform the scan.

    Returns
    -------
    max_coa : `numpy.ndarray', double [nsamp]
        Continuous maximum coalescence values in the 3-D volume.
    max_coa_n : `numpy.ndarray`, double [nsamp]
        Continuous normalised maximum coalescence values in the 3-D volume.
    max_idx : `numpy.ndaarray`, int [nsamp]
        Flattened grid indices of the continuous maximum coalescence values in
        the 3-D volume.

    """

    nsamp = map4d.shape[-1]
    ncell = np.prod(map4d.shape[:-1])
    max_coa = np.zeros(nsamp, np.double)
    max_coa_n = np.zeros(nsamp, np.double)
    max_idx = np.zeros(nsamp, np.int64)

    _qmigratelib.find_max_coa(map4d, max_coa, max_coa_n, max_idx,
                              c_int32(nsamp), c_int64(ncell), c_int64(threads))

    return max_coa, max_coa_n, max_idx
