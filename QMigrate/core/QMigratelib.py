# -*- coding: utf-8 -*-
"""
This module requires that 'numpy' is installed in your Python environment.

This module acts as a wrapper for the C-compiled functions that make up the
core of the QuakeMigrate package.

TO-DO
-----

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
print(str(p))
if os.name == 'nt':
    _qmigratelib = clib.load_library(str(p.with_suffix(".dll")), ".")
else:  # posix
    _qmigratelib = clib.load_library(str(p.with_suffix(".so")), ".")


_qmigratelib.onset.argtypes = [c_dPt, c_int, c_int,
                               c_int, c_int, c_dPt]
_qmigratelib.onset_mp.argtypes = [c_dPt, c_int, c_int, c_int,
                                  c_int, c_int, c_dPt]


def onset(env, stw, ltw, gap):
    """
    Wrapper for the C-compiled onset and onset_mp functions

    Parameters
    ----------
    env : 

    stw : 

    ltw : 

    gap : 

    Returns
    -------
    out : 

    """

    ntr = env[..., 0].size
    nsamp = env.shape[-1]
    out = np.zeros(env.shape, dtype=np.float64)
    env = np.ascontiguousarray(env, np.float64)
    if ntr > 1:
        _qmigratelib.onset_mp(env, ntr, nsamp, int(stw),
                              int(ltw), int(gap), out)
        return out
    else:
        _qmigratelib.onset(env, nsamp, int(stw),
                           int(ltw), int(gap), out)
        return out


_qmigratelib.levinson.argtypes = [c_dPt, c_int, c_dPt,
                                  c_dPt, c_dPt, c_dPt]

_qmigratelib.levinson_mp.argtypes = [c_dPt, c_int, c_int, c_int,
                                     c_dPt, c_dPt, c_dPt, c_dPt]


def levinson(acc, order, return_error=False):
    """
    Wrapper for the C-compiled levinson and levinson_mp functions

    Parameters
    ----------
    acc : 

    order : 

    return_error : bool

    Returns
    -------
    a : 

    k : 

    e : , optional

    """

    acc = np.array(acc, dtype=np.double)
    if acc.ndim > 1:
        nsamp = acc.shape[-1]
        nchan = acc[..., 0].size
        chan = acc.shape[:-1]
        a = np.zeros(chan + (order+1,), dtype=np.double)
        e = np.zeros(chan, dtype=np.double)
        k = np.zeros(chan + (order,), dtype=np.double)
        tmp = np.zeros(chan + (order,), dtype=np.double)
        _qmigratelib.levinson_mp(acc, nchan, nsamp, order, a,
                                 e, k, tmp)
    else:
        nsamp = acc.shape[-1]
        order = min(order, nsamp-1)
        a = np.zeros(order+1, dtype=np.double)
        e = np.zeros(1, dtype=np.double)
        k = np.zeros(order, dtype=np.double)
        tmp = np.zeros(order, dtype=np.double)
        _qmigratelib.levinson(acc, order, a,
                              e, k, tmp)
    if return_error:
        return a, k, e
    else:
        return a, k


_qmigratelib.nlevinson.argtypes = [c_dPt, c_int,
                                   c_dPt, c_dPt]

_qmigratelib.nlevinson_mp.argtypes = [c_dPt, c_int, c_int,
                                      c_dPt, c_dPt]


def nlevinson(acc):
    """
    Wrapper for the C-compiled nlevinson and nlevinson_mp functions

    Parameters
    ----------
    acc : 

    Returns
    -------
    a : 

    """
    acc = np.array(acc, dtype=np.double)
    if acc.ndim > 1:
        nsamp = acc.shape[-1]
        nchan = acc[..., 0].size
        a = np.zeros(acc.shape, dtype=np.double)
        tmp = np.zeros(acc.shape, dtype=np.double)
        _qmigratelib.nlevinson_mp(acc, nchan, nsamp,
                                  a, tmp)
    else:
        nsamp = acc.shape[-1]
        a = np.zeros(nsamp, dtype=np.double)
        tmp = np.zeros(nsamp, dtype=np.double)
        _qmigratelib.nlevinson(acc, nsamp,
                               a, tmp)
    return a


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


# _qmigratelib.detect4d_t.argtypes = [c_dPt, c_dPt, c_i64Pt, c_int32,
#                                    c_int32, c_int32, c_int64, c_int64]

# def detect_t(mmap, dsnr, dind, fsmp, lsmp, threads):
#     """
#     Wrapper for the C-compiled detect4d_t function

#     Parameters
#     ----------
#     mmap :

#     dsnr : 

#     dind : 

#     fsmp : 

#     lsmp : 

#     threads :


#     Raises
#     ------
#     ValueError
#         If the output array size is too small

#     """

#     nsamp = mmap.shape[0]
#     ncell = np.prod(mmap.shape[1:])
#     if dsnr.size < nsamp or dind.size < nsamp:
#         msg = "Output array size is too small, sample count = {}."
#         msg = msg.format(nsamp)
#         raise ValueError(msg)

#     _qmigratelib.detect4d_t(mmap, dsnr, dind, c_int32(fsmp), c_int32(lsmp),
#                            c_int32(nsamp), c_int64(ncell), c_int64(threads))
