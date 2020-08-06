# -*- coding: utf-8 -*-
"""
The default onset function class - performs some pre-processing on raw
seismic data and calculates STA/LTA onset (characteristic) function.

"""

import logging

import numpy as np
from obspy import Stream
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


def pre_process(stream, sampling_rate, resample, upfactor, filter_):
    """
    Detrend raw seismic data and apply cosine taper and zero phase-shift
    Butterworth band-pass filter.

    Parameters
    ----------
    stream : `~obspy.Stream` object
        Data signal to be pre-processed.
    sampling_rate : int
        Number of samples per second, in Hz.
    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly
        to the desired sampling rate.
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to
        the desired sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.
    filter_ : list
        Filter specifications, as [lowcut (Hz), highcut (Hz), order]. NOTE -
        two-pass filter effectively doubles the number of corners.

    Returns
    -------
    filtered_waveforms : `~obspy.Stream` object
        Pre-processed seismic data.

    Raises
    ------
    NyquistException
        If the high-cut filter specified for the bandpass filter is higher than
        the Nyquist frequency of the `Waveform.signal` data.

    """

    # Resample the data here
    resampled_stream = util.resample(stream, sampling_rate, resample, upfactor)

    # Grab filter info
    lowcut, highcut, order = filter_
    # Construct butterworth band-pass filter
    try:
        b1, a1 = butter(order, [2.0 * lowcut / sampling_rate,
                                2.0 * highcut / sampling_rate], btype="band")
    except ValueError:
        raise util.NyquistException(highcut, 0.5 * sampling_rate, "")

    # Construct cosine taper
    taper = cosine_taper(len(resampled_stream[0].data), 0.1)

    filtered_waveforms = Stream()

    # Detrend, apply cosine taper then apply band-pass filter in both
    # directions for zero phase-shift
    for trace in resampled_stream:
        filtered_trace = trace.copy()
        filtered_data = filtered_trace.data
        filtered_data = detrend(filtered_data, type="linear")
        filtered_data = detrend(filtered_data, type="constant")
        filtered_data = filtered_data * taper
        filtered_data = lfilter(b1, a1, filtered_data[::-1])[::-1]
        filtered_data = lfilter(b1, a1, filtered_data)
        filtered_trace.data = filtered_data
        filtered_waveforms.append(filtered_trace)

    return filtered_waveforms


class STALTAOnset(Onset):
    """
    QuakeMigrate default onset function class - uses the Short-Term Average to
    Long-Term Average ratio.

    Attributes
    ----------
    bandpass_filters : dict of [float, float, int]
        Butterworth bandpass filter specification - keys are phases.
        [lowpass (Hz), highpass (Hz), corners*]
        *NOTE: two-pass filter effectively doubles the number of corners.
    channel_maps : dict of str
        Data component maps - keys are phases. These are passed into the ObsPy
        stream.select method.
    onset_windows : dict of [float, float]
        Onset window lengths - keys are phases.
        [STA, LTA] (both in seconds)
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
    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly
        to the desired sampling rate.
    sampling_rate : int
        Desired sampling rate for input data, in Hz; sampling rate at which
        the onset functions will be computed.
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to
        the desired sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.

    Methods
    -------
    calculate_onsets
        Generate onset functions that represent seismic phase arrivals.
    gaussian_halfwidth
        Phase-appropriate Gaussian half-width estimate based on the short-term
        average window length.

    """

    def __init__(self, **kwargs):
        """Instantiate the STALTAOnset object."""

        super().__init__(**kwargs)

        self.position = kwargs.get("position", "classic")
        self.bandpass_filters = kwargs.get("bandpass_filters",
                                           {"P": [2.0, 16.0, 2],
                                            "S": [2.0, 16.0, 2]})
        self.onset_windows = kwargs.get("onset_windows", {"P": [0.2, 1.0],
                                                          "S": [0.2, 1.0]})
        self.channel_maps = kwargs.get("channel_maps", {"P": "*Z",
                                                        "S": "*[N,E]"})
        self.resample = kwargs.get("resample", False)
        self.upfactor = kwargs.get("upfactor")

        # --- Deprecated ---
        self.onset_centred = kwargs.get("onset_centred")
        self.p_bp_filter = kwargs.get("p_bp_filter")
        self.s_bp_filter = kwargs.get("s_bp_filter")
        self.p_onset_win = kwargs.get("p_onset_win")
        self.s_onset_win = kwargs.get("s_onset_win")

    def __str__(self):
        """Return short summary string of the Onset object."""

        out = (f"\tOnset parameters - using the {self.position} STA/LTA onset"
               f"\n\t\tData sampling rate = {self.sampling_rate} Hz\n")
        for phase, filt in self.bandpass_filters.items():
            out += f"\n\t\t{phase} bandpass filter  = {filt} (Hz, Hz, -)"
        out += "\n"
        for phase, windows in self.onset_windows.items():
            out += f"\n\t\t{phase} onset [STA, LTA] = {windows} (s, s)"

        return out

    def calculate_onsets(self, data, phases=["P", "S"], log=True):
        """
        Returns a stacked pair of onset (characteristic) functions for the
        requested phases.

        Parameters
        ----------
        data : :class:`~quakemigrate.io.data.SignalData` object
            Light class encapsulating data returned by an archive query.
        phases : list of str
            Specify the phases to use for the onset calculation.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.

        Returns
        -------
        onsets : `numpy.ndarray` of float
            Stacked onset functions served up to the scanner.
        stations : list of str
            Available stations, used to request the correct traveltime lookup
            tables.

        """

        for phase in phases:
            phase_waveforms = data.waveforms.select(
                channel=self.channel_maps[phase])

            stw, ltw = self.onset_windows[phase]
            stw = util.time2sample(stw, self.sampling_rate) + 1
            ltw = util.time2sample(ltw, self.sampling_rate) + 1

            filtered_waveforms = pre_process(
                phase_waveforms, self.sampling_rate, self.resample,
                self.upfactor, self.bandpass_filters[phase])

            for station in data.stations:
                waveforms = filtered_waveforms.select(station=station)

                # Check for flatlines and any available data
                if any([tr.data.max() == tr.data.min() for tr in waveforms]):
                    continue
                if not bool(waveforms):
                    continue

                data.onsets.setdefault(station, {}).update(
                    {phase: self._onset(waveforms, stw, ltw, log)})

            data.filtered_waveforms += filtered_waveforms

        data.availability = {station: int(len(data.onsets[station]))
                             for station in data.onsets.keys()}

        onsets = []
        for phase in phases:
            for station in data.onsets.keys():
                onsets.append(data.onsets[station][phase])
        return np.stack(onsets, axis=0), data.availability.keys()

    def _onset(self, stream, stw, ltw, log):
        """
        Generates an onset (characteristic) function. If there are multiple
        components, these are combined as the root-mean-square of the onset
        functions BEFORE taking a log (if requested).

        Parameters
        ----------
        stream : `~obspy.Stream` object
            Stream containing the pre-processed data to calculate the onset.
        stw : int
            Number of samples in the short-term window.
        ltw : int
            Number of samples in the long-term window.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.

        Returns
        -------
        onset : `numpy.ndarray` of float
            STA/LTA onset function.

        """

        if self.position == "centred":
            onsets = [sta_lta_centred(tr.data, stw, ltw) for tr in stream]
        elif self.position == "classic":
            onsets = [classic_sta_lta(tr.data, stw, ltw) for tr in stream]

        onset = np.sqrt(np.sum(np.stack([onset ** 2 for onset in onsets]),
                               axis=0) / len(onsets))

        np.clip(1 + onset, 0.8, np.inf, onset)
        if log:
            np.log(onset, onset)

        return onset

    def gaussian_halfwidth(self, phase):
        """
        Return the phase-appropriate Gaussian half-width estimate based on the
        short-term average window length.

        Parameters
        ----------
        phase : {'P', 'S'}
            Seismic phase for which to serve the estimate.

        """

        return self.onset_windows[phase][0] * self.sampling_rate / 2

    @property
    def pre_pad(self):
        """Pre-pad is determined as a function of the onset windows"""
        windows = self.onset_windows
        pre_pad = (max([windows[key][1] for key in windows.keys()])
                   + 3 * max([windows[key][0] for key in windows.keys()]))

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
        windows = self.onset_windows
        lta_max = max([windows[key][1] for key in windows.keys()])
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

    @property
    def p_bp_filter(self):
        """Handle deprecated p_bp_filter kwarg / attribute"""
        return self.bandpass_filters["P"]

    @p_bp_filter.setter
    def p_bp_filter(self, value):
        """
        Handle deprecated p_bp_filter kwarg / attribute and print warning.

        """
        if value is None:
            return
        print("FutureWarning: Parameter name has changed - continuing.")
        print("To remove this message, refer to the documentation.")

        self.bandpass_filters["P"] = value

    @property
    def s_bp_filter(self):
        """Handle deprecated s_bp_filter kwarg / attribute"""
        return self.bandpass_filters["S"]

    @s_bp_filter.setter
    def s_bp_filter(self, value):
        """
        Handle deprecated s_bp_filter kwarg / attribute and print warning.

        """
        if value is None:
            return
        print("FutureWarning: Parameter name has changed - continuing.")
        print("To remove this message, refer to the documentation.")

        self.bandpass_filters["S"] = value

    @property
    def p_onset_win(self):
        """Handle deprecated p_onset_win kwarg / attribute"""
        return self.onset_windows["P"]

    @p_onset_win.setter
    def p_onset_win(self, value):
        """
        Handle deprecated p_onset_win kwarg / attribute and print warning.

        """
        if value is None:
            return
        print("FutureWarning: Parameter name has changed - continuing.")
        print("To remove this message, refer to the documentation.")

        self.onset_windows["P"] = value

    @property
    def s_onset_win(self):
        """Handle deprecated s_onset_win kwarg / attribute"""
        return self.onset_windows["S"]

    @s_onset_win.setter
    def s_onset_win(self, value):
        """
        Handle deprecated s_onset_win kwarg / attribute and print warning.

        """
        if value is None:
            return
        print("FutureWarning: Parameter name has changed - continuing.")
        print("To remove this message, refer to the documentation.")

        self.onset_windows["S"] = value


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
