# -*- coding: utf-8 -*-
"""
Bindings for the QuakeMigrate C libraries.

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import numpy as np
import numpy.ctypeslib as clib

from quakemigrate.core.libnames import _load_cdll
import quakemigrate.util as util


qmlib = _load_cdll("qmlib")

# Make datatype aliases and build custom datatypes
c_int32 = clib.ctypes.c_int32
c_int64 = clib.ctypes.c_int64
c_dPt = clib.ndpointer(dtype=np.double, flags="C_CONTIGUOUS")
c_i32Pt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_i64Pt = clib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")

stalta_header_t = np.dtype(
    [("n", c_int32), ("nsta", c_int32), ("nlta", c_int32)], align=True
)
stalta_header_pt = clib.ndpointer(stalta_header_t, flags="C_CONTIGUOUS")

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
def migrate(
    onsets: np.ndarray[np.double],
    traveltimes: np.ndarray[int],
    first_idx: int,
    last_idx: int,
    available: int,
    threads: int,
) -> np.ndarray[np.double]:
    """
    Computes 4-D coalescence map by migrating seismic phase onset functions.

    Parameters
    ----------
    onsets: Onset functions for each seismic phase, shape(nonsets, nsamples).
    traveltimes: Grids of seismic phase traveltimes, converted to an integer multiple of
        the sampling rate, shape(nx, ny, nz, nonsets).
    first_idx: Index of first sample in array from which to scan.
    last_idx: Index of last sample in array up to which to scan.
    available: Number of available onset functions.
    threads: Number of threads with which to perform the scan.

    Returns
    -------
    map4d: 4-D coalescence map, shape(nx, ny, nz, nsamples).

    Raises
    ------
    ValueError
        If mismatch between number of onset functions and traveltime lookup tables.
        Expect both to be equal to the no. stations * no. phases.
    ValueError
        If the number of samples in the onset functions is less than the number of
        samples array is smaller than map4d[0, 0, 0, :].

    """

    # By taking the log of the onsets, we can calculate the geometric mean as an
    # arithmetic mean (we then exponentiate within the C function to return the
    # correct coalescence value). Clip as a safety check to prevent trying to take
    # log(0).
    onsets = np.clip(onsets, 0.01, np.inf)
    onsets = np.log(onsets)

    *grid_dimensions, n_luts = traveltimes.shape
    n_onsets, t_samples = onsets.shape
    n_samples = t_samples - first_idx - last_idx
    map4d = np.zeros(tuple(grid_dimensions) + (n_samples,), dtype=np.double)
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
def find_max_coa(
    map4d: np.ndarray[np.double], threads: int
) -> tuple[np.ndarray[np.double], np.ndarray[np.double], np.ndarray[int]]:
    """
    Finds time series of the maximum coalescence/normalised coalescence in the 3-D
    volume, and the corresponding grid indices.

    Parameters
    ----------
    map4d: 4-D coalescence map, shape(nx, ny, nz, nsamples).
    threads: Number of threads with which to perform the scan.

    Returns
    -------
    max_coa: Time series of the maximum coalescence value in the 3-D volume.
    max_norm_coa: Times series of the maximum normalised coalescence value in the 3-D
        volume.
    max_coa_idx: Time series of the flattened grid indices corresponding to the maximum
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


qmlib.overlapping_sta_lta.argtypes = [c_dPt, stalta_header_pt, c_dPt]


def overlapping_sta_lta(
    signal: np.ndarray[float], nsta: int, nlta: int
) -> np.ndarray[np.double]:
    """
    Compute the STA/LTA onset function with overlapping windows. The return
    value is allocated to the last sample of the STA window.

                                                 |--- STA ---|
     |------------------------- LTA -------------------------|
                                                             ^
                                                    Value assigned here

    Parameters
    ----------
    signal: Pre-processed waveform data to be processed into an onset function.
    nsta: Number of samples in the short-term average window.
    nlta: Number of samples in the long-term average window.

    Returns
    -------
    onset: Overlapping STA/LTA onset function.

    """

    # Build header structure and ensure signal data is contiguous in memory
    head = np.empty(1, dtype=stalta_header_t)
    head[:] = (len(signal), nsta, nlta)
    signal = np.ascontiguousarray(signal, dtype=np.double)
    onset = np.ones(len(signal), dtype=np.double)

    qmlib.overlapping_sta_lta(signal, head, onset)

    return onset


qmlib.centred_sta_lta.argtypes = [c_dPt, stalta_header_pt, c_dPt]


def centred_sta_lta(
    signal: np.ndarray[float], nsta: int, nlta: int
) -> np.ndarray[np.double]:
    """
    Compute the STA/LTA onset function with consecutive windows. The return
    value is allocated to the last sample of the LTA window.

                                                            |--- STA ---|
         |---------------------- LTA ----------------------|
                                                           ^
                                                  Value assigned here

    Parameters
    ----------
    signal: Pre-processed waveform data to be processed into an onset function.
    nsta: Number of samples in the short-term average window.
    nlta: Number of samples in the long-term average window.

    Returns
    -------
    onset: Centred STA/LTA onset function.

    """

    # Build header structure and ensure signal data is contiguous in memory
    head = np.empty(1, dtype=stalta_header_t)
    head[:] = (len(signal), nsta, nlta)
    signal = np.ascontiguousarray(signal, dtype=np.double)
    onset = np.ones(len(signal), dtype=np.double)

    qmlib.centred_sta_lta(signal, head, onset)

    return onset


qmlib.recursive_sta_lta.argtypes = [c_dPt, stalta_header_pt, c_dPt]


def recursive_sta_lta(
    signal: np.ndarray[float], nsta: int, nlta: int
) -> np.ndarray[np.double]:
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
    signal: Pre-processed waveform data to be processed into an onset function.
    nsta: Number of samples in the short-term average window.
    nlta: Number of samples in the long-term average window.

    Returns
    -------
    onset: Recursive (centred) STA/LTA onset function.

    """

    # Build header structure and ensure signal data is contiguous in memory
    head = np.empty(1, dtype=stalta_header_t)
    head[:] = (len(signal), nsta, nlta)
    signal = np.ascontiguousarray(signal, dtype=np.double)
    onset = np.zeros(len(signal), dtype=np.double)

    qmlib.recursive_sta_lta(signal, head, onset)

    return onset
