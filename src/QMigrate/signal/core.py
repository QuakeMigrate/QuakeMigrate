
import numpy as np
import scipy.signal as ssp
from scipy.signal import butter, lfilter, detrend
from ..core import cmslib


def nextpow2(n):
    """
    Return the next power of 2 such that 2^p >= n.

    :param n: Integer number of samples.
    :return: p
    """

    if np.any(n < 0):
        raise ValueError("n should be > 0")

    if np.isscalar(n):
        f, p = np.frexp(n)
        if f == 0.5:
            return p-1
        elif np.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = np.frexp(n)
        res = f
        bet = np.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res


def detrend(sig, mean=False):
    """

    :param sig: numpy.ndarray
    :param mean: If true subtract the mean otherwise linear trend.
    :return: detrended signal
    """
    if mean:
        return sig - sig.mean(-1)[..., np.newaxis]
    else:
        return ssp.detrend(sig)


def filter(sig, srate, lc, hc, order=3):
    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='bandpass')
    indx = np.zeros((sig.shape[-1],), dtype=np.int)
    fsig = sig - sig[..., indx]
    fsig = lfilter(b1, a1, fsig)
    return fsig


def filtfilt(sig, srate, lc, hc, order=3):
    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='bandpass')
    indx = np.zeros((sig.shape[-1],), dtype=np.int)
    fsig = sig - sig[..., indx]
    fsig = lfilter(b1, a1, fsig[..., ::-1])
    fsig = lfilter(b1, a1, fsig[..., ::-1])
    return fsig


def tmshift(sig, tm, srate=1, taper=None):
    w = -2*np.pi*1j
    #fd = np.iscomplex(sig)
    if sig.ndim == 1:
        nsamp = sig.size
        fsig = np.fft.fft(sig)
        freq = np.fft.fftfreq(nsamp, 1 / srate)
        sft = np.exp(w * freq * tm)
        return np.real(np.fft.ifft(fsig * sft))
    else:
        nsamp = sig.shape[-1]
        fsig = np.fft.fft(sig)
        freq = np.fft.fftfreq(nsamp, 1 / srate)
        if np.isscalar(tm):
            wf = np.exp(w*tm*freq)
        else:
            tm = np.expand_dims(tm, 1)
            wf = np.exp(w*tm*np.expand_dims(freq, 0))
        return np.real(np.fft.ifft(wf * fsig))

def complex_env(sig):
    sig = detrend(sig)
    fft_sig = np.fft.fft(2*sig)
    fft_sig[..., 0] *= 0.5
    ns = fft_sig.shape[-1]
    fft_sig[..., (ns+1)//2:] = 0.0
    return np.fft.ifft(fft_sig)


def onset(sig, stw, ltw, srate=1, gap=0, log=False):
    env = np.abs(complex_env(sig))
    stw = int(stw*srate + 0.5)
    ltw = int(ltw*srate + 0.5)
    snr = cmslib.onset(env, stw, ltw, gap)
    if log:
        np.clip(1+snr, 0.8, np.inf, snr)
        np.log(snr, snr)
    return snr


def onset2(sig1, sig2, stw, ltw, srate=1, gap=0, log=False):
    env1 = np.abs(complex_env(sig1))
    env2 = np.abs(complex_env(sig2))
    env = np.sqrt(np.abs(env1 * env1 + env2 * env2))
    stw = int(stw*srate + 0.5)
    ltw = int(ltw*srate + 0.5)
    snr = cmslib.onset(env, stw, ltw, gap)
    if log:
        np.clip(1+snr, 0.8, np.inf, snr)
        np.log(snr, snr)
    return snr


def onset_p(sig, stw, ltw, srate=1, gap=0, log=True):
    if sig.ndim > 2:
        return onset(sig[2, :], stw, ltw, srate, gap, log)
    else:
        return onset(sig, stw, ltw, srate, gap, log)


def onset_s(sig, stw, ltw, srate=1, gap=0, log=True):
    enx = complex_env(sig[0, :, :])
    eny = complex_env(sig[1, :, :])
    stw = int(stw*srate + 0.5)
    ltw = int(ltw*srate + 0.5)
    env = np.sqrt(np.abs(enx*enx + eny*eny))
    snr = cmslib.onset(env, stw, ltw, gap)
    if log:
        np.clip(1+snr, 0.8, np.inf, snr)
        np.log(snr, snr)
    return snr
