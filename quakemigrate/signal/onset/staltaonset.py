# -*- coding: utf-8 -*-
"""
The default onset function class - performs some pre-processing on raw
seismic data and calculates STA/LTA onset (characteristic) function.

"""

import numpy as np
from obspy.signal.invsim import cosine_taper
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import butter, lfilter, detrend

import quakemigrate.util as util
from .onset import Onset


def sta_lta_centred(a, nsta, nlta):
    """
    Calculates the ratio of the average signal in a short-term (signal) window
    to a preceding long-term (noise) window. STA/LTA value is assigned to the
    end of the LTA / start of the STA.

    Parameters
    ----------
    a : array-like
        Signal array
    nsta : int
        Number of samples in short-term window
    nlta : int
        Number of samples in long-term window

    Returns
    -------
    sta / lta : array-like
        Ratio of short term average window to a preceding long term average
        window. STA/LTA value is assigned to end of LTA window / start of STA
        window -- "centred"

    """

    # Cumulative sum to calculate moving average
    sta = np.cumsum(a ** 2)
    sta = np.require(sta, dtype=np.float)
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta[nsta:-nsta] = sta[nsta*2:]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    sta[:(nlta - 1)] = 0
    sta[-nsta:] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(float).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = 0.0

    return sta / lta


def sta_lta_onset(fsig, stw, ltw, position, log):
    """
    Calculate STA/LTA onset (characteristic) function from pre-processed
    seismic data.

    Parameters
    ----------
    fsig : array-like
        Filtered (pre-processed) data signal to be used to generate an onset
        function.
    stw : int
        Short term window length (# of samples).
    ltw : int
        Long term window length (# of samples)
    position : str
        "centred": Compute centred STA/LTA (STA window is preceded by LTA
            window; value is assigned to end of LTA window / start of STA
            window) or:
        "classic": classic STA/LTA (STA window is within LTA window; value
            is assigned to end of STA & LTA windows).

        Centred gives less phase-shifted (late) onset function, and is closer
        to a Gaussian approximation, but is far more sensitive to data with
        sharp offsets due to instrument failures. We recommend using classic
        for detect() and centred for locate() if your data quality allows it.
        This is the default behaviour; override by setting self.onset_centred.
    log : bool
        Will return log(onset) if True, otherwise it will return the raw onset.

    Returns
    -------
    onset : array-like
        onset_raw or log(onset_raw); both are clipped between 0.8 and
        infinity.

    """

    n_channels, _ = fsig.shape
    onset = np.copy(fsig)
    for i in range(n_channels):
        if np.sum(fsig[i, :]) == 0.0:
            onset[i, :] = 0.0
        else:
            if position == "centred":
                onset[i, :] = sta_lta_centred(fsig[i, :], stw, ltw)
            elif position == "classic":
                onset[i, :] = classic_sta_lta(fsig[i, :], stw, ltw)
            np.clip(1 + onset[i, :], 0.8, np.inf, onset[i, :])
            if log:
                np.log(onset[i, :], onset[i, :])

    return onset


def pre_process(sig, sampling_rate, lc, hc, order=2):
    """
    Detrend raw seismic data and apply cosine taper and zero phase-shift
    Butterworth band-pass filter.

    Parameters
    ----------
    sig : array-like
        Data signal to be pre-processed.
    sampling_rate : int
        Number of samples per second, in Hz.
    lc : float
        Lowpass frequency of band-pass filter, in Hz.
    hc : float
        Highpass frequency of band-pass filter, in Hz.
    order : int, optional
        Number of filter corners. NOTE: two-pass filter effectively doubles the
        number of corners.

    Returns
    -------
    fsig : array-like
        Filtered seismic data.

    Raises
    ------
    NyquistException
        If the high-cut filter specified for the bandpass filter is higher than
        the Nyquist frequency of the `Waveform.signal` data.

    """

    # Construct butterworth band-pass filter
    try:
        b1, a1 = butter(order, [2.0 * lc / sampling_rate,
                                2.0 * hc / sampling_rate], btype="band")
    except ValueError:
        raise util.NyquistException(hc, 0.5 * sampling_rate, "")

    # Construct cosine taper
    tap = cosine_taper(len(sig[0, :]), 0.1)

    nchan, _ = sig.shape
    fsig = np.copy(sig)

    # Detrend, apply cosine taper then apply band-pass filter in both
    # directions for zero phase-shift
    for ch in range(0, nchan):
        fsig[ch, :] = detrend(fsig[ch, :], type='linear')
        fsig[ch, :] = detrend(fsig[ch, :], type='constant')
        fsig[ch, :] = fsig[ch, :] * tap
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, ::-1])[::-1]
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, :])

    return fsig


class STALTAOnset(Onset):
    """
    QuakeMigrate default onset function class - uses a classic STA/LTA onset.

    Attributes
    ----------
    p_bp_filter : array-like, [float, float, int]
        Butterworth bandpass filter specification
        [lowpass (Hz), highpass (Hz), corners*]
        *NOTE: two-pass filter effectively doubles the number of corners.
    s_bp_filter : array-like, [float, float, int]
        Butterworth bandpass filter specification
        [lowpass (Hz), highpass (Hz), corners*]
        *NOTE: two-pass filter effectively doubles the number of corners.
    p_onset_win : array-like, [float, float]
        P onset window parameters
        [STA, LTA] (both in seconds)
    s_onset_win : array-like, [float, float]
        S onset window parameters
        [STA, LTA] (both in seconds)
    sampling_rate : int
        Desired sampling rate for input data, in Hz; sampling rate at which
        the onset functions will be computed.
    pre_pad : float, optional
        Option to override the default pre-pad duration of data to read
        before computing 4-D coalescence in detect() and locate(). Default
        value is calculated from the onset function parameters.
    position : str, optional
        Compute centred STA/LTA (STA window is preceded by LTA window;
        value is assigned to end of LTA window / start of STA window) or
        classic STA/LTA (STA window is within LTA window; value is assigned
        to end of STA & LTA windows). Default: "classic".

        Centred gives less phase-shifted (late) onset function, and is
        closer to a Gaussian approximation, but is far more sensitive to
        data with sharp offsets due to instrument failures. We recommend
        using classic for detect() and centred for locate() if your data
        quality allows it. This is the default behaviour; override by
        setting this variable.

    Methods
    -------
    calculate_onsets()
        Generate onset functions that represent seismic phase arrivals

    """

    def __init__(self, **kwargs):
        """Instantiate the STALTAOnset object."""

        super().__init__(**kwargs)

        self.position = kwargs.get("position", "classic")
        self.onset_centred = kwargs.get("onset_centred")
        self.p_bp_filter = kwargs.get("p_bp_filter", [2.0, 16.0, 2])
        self.p_onset_win = kwargs.get("p_onset_win", [0.2, 1.0])
        self.s_bp_filter = kwargs.get("s_bp_filter", [2.0, 12.0, 2])
        self.s_onset_win = kwargs.get("s_onset_win", [0.2, 1.0])

    def __str__(self):
        """Return short summary string of the Onset object."""

        out = (f"\tOnset parameters - using the {self.position} STA/LTA onset"
               f"\n\t\tData sampling rate = {self.sampling_rate} Hz\n"
               f"\n\t\tBandpass filter P  = {self.p_bp_filter} (Hz, Hz, -)"
               f"\n\t\tBandpass filter S  = {self.s_bp_filter} (Hz, Hz, -)\n"
               f"\n\t\tOnset P [STA, LTA] = {self.p_onset_win} (s, s)"
               f"\n\t\tOnset S [STA, LTA] = {self.s_onset_win} (s, s)\n")

        return out

    def calculate_onsets(self, data, log=True, run=None):
        """
        Returns a stacked pair of onset (characteristic) functions for the P
        and S phase arrivals.

        Parameters
        ----------
        data : :class:`~quakemigrate.io.data.SignalData` object
            Light class encapsulating data returned by an archive query.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.
        run :

        """

        filtered_signal = np.empty((data.signal.shape))
        filtered_signal[:] = np.nan

        p_onset, filtered_signal[2, :, :] = self._p_onset(data.signal[2], log)
        s_onset, filtered_signal[1, :, :], \
            filtered_signal[0, :, :] = self._s_onset(data.signal[0],
                                                     data.signal[1], log)
        if not (isinstance(p_onset, np.ndarray)
                and isinstance(s_onset, np.ndarray)):
            raise TypeError
        data.p_onset = p_onset
        data.s_onset = s_onset

        ps_onset = np.concatenate((p_onset, s_onset))
        ps_onset[np.isnan(ps_onset)] = 0

        data.filtered_signal = filtered_signal

        return ps_onset

    def _p_onset(self, sigz, log):
        """
        Generates an onset (characteristic) function for the P-phase from the
        Z-component signal.

        Parameters
        ----------
        sigz : array-like
            Z-component time series.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.

        Returns
        -------
        p_onset : array-like
            Onset function generated from STA/LTA of vertical component data.
        filt_sigz : array-like
            Pre-processed vertical component data (detrended, tapered and
            bandpass filtered.)

        """

        stw, ltw = self.p_onset_win
        stw = util.time2sample(stw, self.sampling_rate) + 1
        ltw = util.time2sample(ltw, self.sampling_rate) + 1

        lc, hc, ord_ = self.p_bp_filter
        filt_sigz = pre_process(sigz, self.sampling_rate, lc, hc, ord_)

        p_onset = sta_lta_onset(filt_sigz, stw, ltw, position=self.position,
                                log=log)

        return p_onset, filt_sigz

    def _s_onset(self, sige, sign, log):
        """
        Generates an onset (characteristic) function for the S-phase from the
        E- and N-components signal.

        Parameters
        ----------
        sige : array-like
            E-component time series.
        sign : array-like
            N-component time series.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.

        Returns
        -------
        s_onset : array-like
            Onset function generated from STA/LTA of horizontal component data.
        filt_sige : array-like
            Pre-processed East-component data (detrended, tapered and
            bandpass filtered.)
        filt_sign : array-like
            Pre-processed North-component data (detrended, tapered and
            bandpass filtered.)

        """

        stw, ltw = self.s_onset_win
        stw = util.time2sample(stw, self.sampling_rate) + 1
        ltw = util.time2sample(ltw, self.sampling_rate) + 1

        lc, hc, ord_ = self.s_bp_filter
        filt_sige = pre_process(sige, self.sampling_rate, lc, hc, ord_)
        filt_sign = pre_process(sign, self.sampling_rate, lc, hc, ord_)

        s_e_onset = sta_lta_onset(filt_sige, stw, ltw, position=self.position,
                                  log=log)
        s_n_onset = sta_lta_onset(filt_sign, stw, ltw, position=self.position,
                                  log=log)

        s_onset = np.sqrt((s_e_onset ** 2 + s_n_onset ** 2) / 2.)

        return s_onset, filt_sige, filt_sign

    def gaussian_halfwidth(self, phase):
        """
        Return the phase-appropriate Gaussian half-width estimate based on the
        short-term average window length.

        Parameters
        ----------
        phase : {'P', 'S'}
            Seismic phase for which to serve the estimate.

        """

        if phase == "P":
            sta_window = self.p_onset_win[0]
        elif phase == "S":
            sta_window = self.s_onset_win[0]

        return (sta_window * self.sampling_rate / 2)

    @property
    def pre_pad(self):
        """Pre-pad is determined as a function of the onset windows"""
        pre_pad = max(self.p_onset_win[1], self.s_onset_win[1]) \
            + 3 * max(self.p_onset_win[0], self.s_onset_win[0])

        return pre_pad

    @pre_pad.setter
    def pre_pad(self, value):
        """Setter for pre-pad"""

        self._pre_pad = value

    @property
    def post_pad(self):
        """
        Post-pad is determined as a function of the max traveltime in the
        grid and the onset windows

        """

        return self._post_pad

    @post_pad.setter
    def post_pad(self, ttmax):
        """
        Define post-pad as a function of the maximum travel-time between a
        station and a grid point plus the LTA (in case onset_centred is True)

        """

        lta_max = max(self.p_onset_win[1], self.s_onset_win[1])
        self._post_pad = np.ceil(ttmax + 2 * lta_max)

    # --- Deprecation/Future handling ---
    @property
    def onset_centred(self):
        """Handle deprecated onset_centred kwarg / attribute"""
        return self.position

    @onset_centred.setter
    def onset_centred(self, value):
        """
        Handle deprecated onset_centred kwarg / attribute and print warning.

        """
        if value is None:
            return
        print("FutureWarning: Parameter name has changed - continuing.")
        print("To remove this message, change:")
        print("\t'onset_centred' -> 'position'")
        if value:
            self.position = "centred"
        else:
            self.position = "classic"


class CentredSTALTAOnset(STALTAOnset):
    """
    QuakeMigrate default onset function class - uses a centred STA/LTA onset.

    NOTE: THIS CLASS HAS BEEN DEPRECATED AND WILL BE REMOVED IN A FUTURE UPDATE

    """

    def __init__(self, **kwargs):
        """Instantiate CentredSTALTAOnset object."""

        super().__init__(**kwargs)

        print("FutureWarning: This class has been deprecated - continuing.")
        print("To remove this message:")
        print("\tCentredSTALTAOnset -> STALTAOnset\n")
        print("\tAnd add keyword argument 'position=centred'")
        self.position = "centred"


class ClassicSTALTAOnset(STALTAOnset):
    """
    QuakeMigrate default onset function class - uses a classic STA/LTA onset.

    NOTE: THIS CLASS HAS BEEN DEPRECATED AND WILL BE REMOVED IN A FUTURE UPDATE

    """

    def __init__(self, **kwargs):
        """Instantiate ClassicSTALTAOnset object."""

        super().__init__(**kwargs)

        print("FutureWarning: This class has been deprecated - continuing.")
        print("To remove this message:")
        print("\tClassicSTALTAOnset -> STALTAOnset")
        print("\tAnd add keyword argument 'position=classic'\n")
        self.position = "classic"
