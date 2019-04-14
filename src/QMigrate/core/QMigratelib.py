
import os
import numpy as np
import numpy.ctypeslib as clib

c_int = clib.ctypes.c_int
c_int8 = clib.ctypes.c_int8
c_int16 = clib.ctypes.c_int16
c_int32 = clib.ctypes.c_int32
c_int64 = clib.ctypes.c_int64
c_dbl = clib.ctypes.c_double
c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_i8Pt = clib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
c_i16Pt = clib.ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")
c_i32Pt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_i64Pt = clib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
c_iPt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

if os.name == 'nt':
    _seisloclib = clib.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/QMigrate.dll'), '.')
else:  # posix
    _seisloclib = clib.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/QMigrate.so'), '.')

_seisloclib.onset.argtypes = [c_dPt, c_int, c_int, c_int, c_int, c_dPt]
_seisloclib.onset_mp.argtypes = [c_dPt, c_int, c_int, c_int, c_int, c_int, c_dPt]

def onset(env, stw, ltw, gap):
    ntr = env[..., 0].size
    nsamp = env.shape[-1]
    out = np.zeros(env.shape, dtype=np.float64)
    env = np.ascontiguousarray(env, np.float64)
    if ntr > 1:
        _seisloclib.onset_mp(env, ntr, nsamp, int(stw), int(ltw), int(gap), out)
        return out
    else:
        _seisloclib.onset(env, nsamp, int(stw), int(ltw), int(gap), out)
        return out


_seisloclib.levinson.argtypes = [c_dPt, c_int, c_dPt, c_dPt, c_dPt, c_dPt]
_seisloclib.levinson_mp.argtypes = [c_dPt, c_int, c_int, c_int, c_dPt, c_dPt, c_dPt, c_dPt]
_seisloclib.nlevinson.argtypes = [c_dPt, c_int, c_dPt, c_dPt]
_seisloclib.nlevinson_mp.argtypes = [c_dPt, c_int, c_int, c_dPt, c_dPt]


def levinson(acc, order, return_error=False):
    acc = np.array(acc, dtype=np.double)
    if acc.ndim > 1:
        nsamp = acc.shape[-1]
        nchan = acc[..., 0].size
        chan = acc.shape[:-1]
        a = np.zeros(chan + (order+1,), dtype=np.double)
        e = np.zeros(chan, dtype=np.double)
        k = np.zeros(chan + (order,), dtype=np.double)
        tmp = np.zeros(chan + (order,), dtype=np.double)
        _seisloclib.levinson_mp(acc, nchan, nsamp, order, a,
                          e, k, tmp)
    else:
        nsamp = acc.shape[-1]
        order = min(order, nsamp-1)
        a = np.zeros(order+1, dtype=np.double)
        e = np.zeros(1, dtype=np.double)
        k = np.zeros(order, dtype=np.double)
        tmp = np.zeros(order, dtype=np.double)
        _seisloclib.levinson(acc, order, a,
                       e, k, tmp)
    if return_error:
        return a, k, e
    else:
        return a, k


def nlevinson(acc):
    acc = np.array(acc, dtype=np.double)
    if acc.ndim > 1:
        nsamp = acc.shape[-1]
        nchan = acc[..., 0].size
        a = np.zeros(acc.shape, dtype=np.double)
        tmp = np.zeros(acc.shape, dtype=np.double)
        _seisloclib.nlevinson_mp(acc, nchan, nsamp,
                           a, tmp)
    else:
        nsamp = acc.shape[-1]
        a = np.zeros(nsamp, dtype=np.double)
        tmp = np.zeros(nsamp, dtype=np.double)
        _seisloclib.nlevinson(acc, nsamp,
                        a, tmp)
    return a




_seisloclib.scan4d.argtypes = [c_dPt,  c_i32Pt, c_dPt, c_int32,  c_int32, c_int32, c_int32, c_int64, c_int64]
_seisloclib.detect4d.argtypes = [c_dPt, c_dPt, c_i64Pt,c_int32, c_int32, c_int32, c_int64, c_int64]
# _seisloclib.detect4d_t.argtypes = [c_dPt, c_dPt, c_i64Pt,c_int32, c_int32, c_int32, c_int64, c_int64]

def scan(sig, tt, fsmp,lsmp, nsamp, map4d, threads):
    nstn, ssmp = sig.shape
    if not tt.shape[-1] == nstn:
        raise ValueError('Mismatch between number of stations for data and LUT, {} - {}.'.format(
            nstn, tt.shape[-1]))
    ncell = tt.shape[:-1]
    tcell = np.prod(ncell)
    if map4d.size < nsamp*tcell:
        raise ValueError('4D-Array is too small.')

    if sig.size < nsamp + fsmp:
        raise ValueError('Data array smaller than Coalescence array')


    _seisloclib.scan4d(sig, tt, map4d, c_int32(fsmp), c_int32(lsmp),c_int32(nsamp), c_int32(nstn),  c_int64(tcell), c_int64(threads))


def detect(mmap, dsnr, dind, fsmp, lsmp,threads):
    nsamp = mmap.shape[-1]
    ncell = np.prod(mmap.shape[:-1])
    if dsnr.size < nsamp or dind.size < nsamp:
        raise ValueError('Ouput array size too small, sample count = {}.'.format(nsamp))
    _seisloclib.detect4d(mmap, dsnr, dind, c_int32(fsmp),c_int32(lsmp),c_int32(nsamp), c_int64(ncell), c_int64(threads))


# def detect_t(mmap, dsnr, dind, fsmp, lsmp, threads):
#     nsamp = mmap.shape[0]
#     ncell = np.prod(mmap.shape[1:])
#     if dsnr.size < nsamp or dind.size < nsamp:
#         raise ValueError('Ouput array size too small, sample count = {}.'.format(nsamp))
#     _seisloclib.detect4d_t(mmap, dsnr, dind, c_int32(fsmp), c_int32(lsmp),
#                      c_int32(nsamp), c_int64(ncell), c_int64(threads))

