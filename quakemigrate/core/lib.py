# -*- coding: utf-8 -*-
"""
Bindings for the C library functions, migrate and find_max_coa.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import numpy as np
import numpy.ctypeslib as clib

from quakemigrate.core.libnames import _load_cdll
import quakemigrate.util as util


qmlib = _load_cdll("qmlib")

c_int32 = clib.ctypes.c_int32
c_int64 = clib.ctypes.c_int64
c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_i32Pt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_i64Pt = clib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")


qmlib.migrate.argtypes = [
    c_dPt,
    c_i32Pt,
    c_dPt,
    c_int32,
    c_int32,
    c_int32,
    c_int32,
    c_int32,
    c_int64,
    c_int64,
]


@util.timeit()
def migrate(onsets, traveltimes, first_idx, last_idx, available, threads):
    """
    Computes 4-D coalescence map by migrating seismic phase onset functions.

    Parameters
    ----------
    onsets : `numpy.ndarry` of float
        Onset functions for each seismic phase, shape(nonsets, nsamples).
    traveltimes : `numpy.ndarry` of int
        Grids of seismic phase traveltimes, converted to an integer multiple of the
        sampling rate, shape(nx, ny, nz, nonsets).
    first_idx : int
        Index of first sample in array from which to scan.
    last_idx : int
        Index of last sample in array up to which to scan.
    available : int
        Number of available onset functions.
    threads : int
        Number of threads with which to perform the scan.

    Returns
    -------
    map4d : `numpy.ndarray` of `numpy.double`
        4-D coalescence map, shape(nx, ny, nz, nsamples).

    Raises
    ------
    ValueError
        If mismatch between number of onset functions and traveltime lookup tables.
        Expect both to be equal to the no. stations * no. phases.
    ValueError
        If the number of samples in the onset functions is less than the number of
        samples array is smaller than map4d[0, 0, 0, :].

    """

    *grid_dimensions, n_luts = traveltimes.shape
    n_onsets, t_samples = onsets.shape
    n_samples = t_samples - first_idx - last_idx
    map4d = np.zeros(tuple(grid_dimensions) + (n_samples,), dtype=np.float64)
    n_nodes = np.prod(grid_dimensions)

    if not n_luts == n_onsets:
        raise ValueError(
            f"Mismatch between number of stations for data and LUT, {n_onsets}:{n_luts}"
        )
    if onsets.size < n_samples + first_idx:
        raise ValueError("Data array smaller than coalescence array.")

    qmlib.migrate(
        onsets,
        traveltimes,
        map4d,
        c_int32(first_idx),
        c_int32(last_idx),
        c_int32(n_samples),
        c_int32(n_onsets),
        c_int32(available),
        c_int64(n_nodes),
        c_int64(threads),
    )

    return map4d


qmlib.find_max_coa.argtypes = [c_dPt, c_dPt, c_dPt, c_i64Pt, c_int32, c_int64, c_int64]


@util.timeit()
def find_max_coa(map4d, threads):
    """
    Finds time series of the maximum coalescence/normalised coalescence in the 3-D
    volume, and the corresponding grid indices.

    Parameters
    ----------
    map4d : `numpy.ndarray` of `numpy.double`
        4-D coalescence map, shape(nx, ny, nz, nsamples).
    threads : int
        Number of threads with which to perform the scan.

    Returns
    -------
    max_coa : `numpy.ndarray` of `numpy.double`
        Time series of the maximum coalescence value in the 3-D volume.
    max_norm_coa : `numpy.ndarray` of `numpy.double`
        Times series of the maximum normalised coalescence value in the 3-D volume.
    max_coa_idx : `numpy.ndarray` of int
        Time series of the flattened grid indices corresponding to the maximum
        coalescence value in the 3-D volume.

    """

    *grid_dimensions, n_samples = map4d.shape
    n_nodes = np.prod(grid_dimensions)
    max_coa = np.zeros(n_samples, dtype=np.double)
    max_norm_coa = np.zeros(n_samples, dtype=np.double)
    max_coa_idx = np.zeros(n_samples, dtype=np.int64)

    qmlib.find_max_coa(
        map4d,
        max_coa,
        max_norm_coa,
        max_coa_idx,
        c_int32(n_samples),
        c_int64(n_nodes),
        c_int64(threads),
    )

    return max_coa, max_norm_coa, max_coa_idx
