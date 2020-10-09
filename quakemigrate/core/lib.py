# -*- coding: utf-8 -*-
"""
Bindings for the QuakeMigrate C library functions.

:copyright:
    2020, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import numpy as np
import numpy.ctypeslib as clib

from quakemigrate.core.libnames import _load_cdll
import quakemigrate.util as util


qmlib = _load_cdll("qmlib")


# Make datatype aliases and build custom datatypes
c_int32 = clib.ctypes.c_int32
c_int64 = clib.ctypes.c_int64
c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_i32Pt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_i64Pt = clib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
stalta_header_t = np.dtype([("n", c_int32),
                            ("nsta", c_int32),
                            ("nlta", c_int32)],
                           align=True)
stalta_header_pt = clib.ndpointer(stalta_header_t, flags="C_CONTIGUOUS")


qmlib.migrate.argtypes = [c_dPt, c_i32Pt, c_dPt, c_int32, c_int32, c_int32,
                          c_int32, c_int32, c_int64, c_int32]


@util.timeit()
def migrate(onsets, traveltimes, first_idx, last_idx, available, threads):
    """
    Computes 4-D coalescence map by migrating seismic phase onset functions.

    Parameters
    ----------
    onsets : `numpy.ndarry` of float
        Onset functions for each seismic phase, shape(nstations, nsamples).
    traveltimes : `numpy.ndarry` of int
        Grids of seismic phase traveltimes converted to an integer multiple of
        the sampling rate, shape(nx, ny, nz, nstations).
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
        If mismatch between number of onset functions and traveltime lookup
        tables - expect both to be equal to the no. stations * no. phases.
    ValueError
        If the number of samples in the onset functions is less than the number
        of samples  array is smaller than map4d[0, 0, 0, :].

    """

    *grid_dimensions, n_luts = traveltimes.shape
    n_onsets, t_samples = onsets.shape
    n_samples = t_samples - first_idx - last_idx
    map4d = np.zeros(tuple(grid_dimensions) + (n_samples,), dtype=np.double)
    n_nodes = np.prod(grid_dimensions)

    if not n_luts == n_onsets:
        raise ValueError("Mismatch between number of stations for data and "
                         f"LUT, {n_onsets} - {n_luts}")
    if onsets.size < n_samples + first_idx:
        raise ValueError("Data array smaller than coalescence array.")

    qmlib.migrate(onsets, traveltimes, map4d, c_int32(first_idx),
                  c_int32(last_idx), c_int32(n_samples), c_int32(n_onsets),
                  c_int32(available), c_int64(n_nodes), c_int32(threads))

    return map4d


qmlib.find_max_coa.argtypes = [c_dPt, c_dPt, c_dPt, c_i64Pt, c_int32, c_int64,
                               c_int32]


@util.timeit()
def find_max_coa(map4d, threads):
    """
    Finds time series of the maximum coalescence/normalised coalescence in the
    3-D volume, and the corresponding grid indices.

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
        Times series of the maximum normalised coalescence value in the 3-D
        volume.
    max_coa_idx : `numpy.ndarray` of int
        Time series of the flattened grid indices corresponding to the maximum
        coalescence value in the 3-D volume.

    """

    *grid_dimensions, n_samples = map4d.shape
    n_nodes = np.prod(grid_dimensions)
    max_coa = np.zeros(n_samples, dtype=np.double)
    max_norm_coa = np.zeros(n_samples, dtype=np.double)
    max_coa_idx = np.zeros(n_samples, dtype=np.int64)

    qmlib.find_max_coa(map4d, max_coa, max_norm_coa, max_coa_idx,
                       c_int32(n_samples), c_int64(n_nodes), c_int32(threads))

    return max_coa, max_norm_coa, max_coa_idx


qmlib.overlapping_sta_lta.argtypes = [c_dPt, stalta_header_pt, c_dPt]


def overlapping_sta_lta(signal, nsta, nlta):
    """
    Compute the STA/LTA onset function with overlapping windows. The return
    value is allocated to the last sample of the STA window.

                                                 |--- STA ---|
     |------------------------- LTA -------------------------|
                                                             ^
                                                    Value assigned here

    Parameters
    ----------
    signal : `numpy.ndarray` of float
        Pre-processed waveform data to be processed into an onset function.
    nsta : int
        Number of samples in the short-term average window.
    nlta : int
        Number of samples in the long-term average window.

    Returns
    -------
    onset : `numpy.ndarray` of `numpy.double`
        Overlapping STA/LTA onset function.

    """

    # Build header structure and ensure signal data is contiguous in memory
    head = np.empty(1, dtype=stalta_header_t)
    head[:] = (len(signal), nsta, nlta)
    signal = np.ascontiguousarray(signal, dtype=np.double)
    onset = np.zeros(len(signal), dtype=np.double)

    qmlib.overlapping_sta_lta(signal, head, onset)

    return onset


qmlib.centred_sta_lta.argtypes = [c_dPt, stalta_header_pt, c_dPt]


def centred_sta_lta(signal, nsta, nlta):
    """
    Compute the STA/LTA onset function with consecutive windows. The return
    value is allocated to the last sample of the LTA window.

                                                            |--- STA ---|
         |---------------------- LTA ----------------------|
                                                           ^
                                                  Value assigned here

    Parameters
    ----------
    signal : `numpy.ndarray` of float
        Pre-processed waveform data to be processed into an onset function.
    nsta : int
        Number of samples in the short-term average window.
    nlta : int
        Number of samples in the long-term average window.

    Returns
    -------
    onset : `numpy.ndarray` of `numpy.double`
        Centred STA/LTA onset function.

    """

    # Build header structure and ensure signal data is contiguous in memory
    head = np.empty(1, dtype=stalta_header_t)
    head[:] = (len(signal), nsta, nlta)
    signal = np.ascontiguousarray(signal, dtype=np.double)
    onset = np.zeros(len(signal), dtype=np.double)

    qmlib.centred_sta_lta(signal, head, onset)

    return onset


qmlib.recursive_sta_lta.argtypes = [c_dPt, stalta_header_pt, c_dPt]


def recursive_sta_lta(signal, nsta, nlta):
    """
    Compute the STA/LTA onset function with consecutive windows using a
    recursive method (minimises memory costs). Reproduces exactly the centred
    STA/LTA onset.

                                                           |--- STA ---|
         |---------------------- LTA ----------------------|
                                                           ^
                                                  Value assigned here

    Parameters
    ----------
    signal : `numpy.ndarray` of float
        Pre-processed waveform data to be processed into an onset function.
    nsta : int
        Number of samples in the short-term average window.
    nlta : int
        Number of samples in the long-term average window.

    Returns
    -------
    onset : `numpy.ndarray` of `numpy.double`
        Recursive (centred) STA/LTA onset function.

    """

    # Build header structure and ensure signal data is contiguous in memory
    head = np.empty(1, dtype=stalta_header_t)
    head[:] = (len(signal), nsta, nlta)
    signal = np.ascontiguousarray(signal, dtype=np.double)
    onset = np.zeros(len(signal), dtype=np.double)

    qmlib.recursive_sta_lta(signal, head, onset)

    return onset
