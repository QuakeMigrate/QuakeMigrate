# -*- coding: utf-8 -*-
"""
The default onset function class - performs some pre-processing on raw seismic data and
calculates STA/LTA onset (characteristic) function.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import numpy as np
from obspy import Stream
from obspy.signal.trigger import classic_sta_lta

import quakemigrate.util as util
from .base import Onset, OnsetData


def sta_lta_centred(signal, nsta, nlta):
    """
    Calculates the ratio of the average of a^2 in a short-term (signal) window to a
    preceding long-term (noise) window. STA/LTA value is assigned to the end of the LTA
    /one sample before the start of the STA.

    Parameters
    ----------
    signal : array-like
        Signal array
    nsta : int
        Number of samples in short-term window
    nlta : int
        Number of samples in long-term window

    Returns
    -------
    sta / lta : array-like
        Ratio of a^2 in a short term average window to a preceding long term average
        window. STA/LTA value is assigned to end of LTA window / one sample before the
        start of STA window -- "centred".

    """

    # Cumulative sum to calculate moving average
    sta = np.cumsum(signal**2)
    sta = np.require(sta, dtype=float)
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta[nsta:-nsta] = sta[nsta * 2 :]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    sta[: (nlta - 1)] = 0
    sta[-nsta:] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(float).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = 0.0

    return sta / lta


def pre_process(stream, sampling_rate, resample, upfactor, filter_, starttime, endtime):
    """
    Resample raw seismic data, detrend and apply cosine taper and zero phase-shift
    Butterworth band-pass filter; all carried out using the built-in obspy functions.

    By default, data with mismatched sampling rates will only be decimated. If
    necessary, and if the user has specified `resample = True` and an `upfactor` to
    upsample by `upfactor = int` for the waveform archive, data can also be upsampled
    and then, if necessary, subsequently decimated to achieve the desired sampling rate.

    For example, for raw input data sampled at a mix of 40, 50 and 100 Hz, to achieve a
    unified sampling rate of 50 Hz, the user would have to specify an `upfactor` of 5;
    40 Hz x 5 = 200 Hz, which can then be decimated to 50 Hz.

    NOTE: data will be detrended and a cosine taper applied before decimation, in order
    to avoid edge effects when applying the lowpass filter.
    See :func:`~quakemigrate.util.resample`

    Parameters
    ----------
    stream : `obspy.Stream` object
        Waveform data to be pre-processed.
    sampling_rate : int
        Desired sampling rate for data to be used to calculate onset. This will be
        achieved by resampling the raw waveform data. By default, only decimation will
        be applied, but data can also be upsampled if specified by the user when
        creating the :class:`~quakemigrate.io.data.Archive` object.
    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    filter_ : list
        Filter specifications, as [lowcut (Hz), highcut (Hz), order]. NOTE - two-pass
        filter effectively doubles the number of corners (order).

    Returns
    -------
    filtered_waveforms : `obspy.Stream` object
        Pre-processed seismic data.

    Raises
    ------
    NyquistException
        If the high-cut filter specified for the bandpass filter is higher than the
        Nyquist frequency of the `sampling_rate`.

    """

    logging.debug(stream.__str__(extended=True))
    logging.debug(f"Resample={resample}, Upfactor={upfactor}")
    # Resample the data here
    resampled_stream = util.resample(
        stream, sampling_rate, resample, upfactor, starttime, endtime
    )

    # Grab filter info
    lowcut, highcut, order = filter_
    # Check that the filter is compatible with the sampling rate
    if highcut >= 0.5 * sampling_rate:
        raise util.NyquistException(highcut, 0.5 * sampling_rate, "")

    # Detrend, apply cosine taper then apply zero-phase band-pass filter
    # Copy to not operate in-place on the input stream
    filtered_waveforms = resampled_stream.copy()
    filtered_waveforms.detrend("linear")
    filtered_waveforms.detrend("constant")
    filtered_waveforms.taper(type="cosine", max_percentage=0.05)
    filtered_waveforms.filter(
        type="bandpass", freqmin=lowcut, freqmax=highcut, corners=order, zerophase=True
    )

    return filtered_waveforms


class STALTAOnset(Onset):
    """
    QuakeMigrate default onset function class - uses the Short-Term Average to Long-Term
    Average ratio of the signal energy amplitude.

    Raw seismic data will be pre-processed, including re-sampling if necessary to reach
    the specified uniform sampling raate, checked against a user- specified set of data
    quality criteria, then used to calculate onset functions for each phase (using
    seismic channels as specified in `channel_maps`) by computing the STA/LTA of s^2.

    Attributes
    ----------
    phases : list of str
        Which phases to calculate onset functions for. This will determine which phases
        are used for migration/picking. The selected phases must be present in the
        travel-time look-up table to be used for these purposes.
    bandpass_filters : dict of [float, float, int]
        Butterworth bandpass filter specification - keys are phases.
        [lowpass (Hz), highpass (Hz), corners*]
        *NOTE: two-pass filter effectively doubles the number of corners.
    channel_maps : dict of str
        Data component maps - keys are phases. These are passed into the
        :meth:`ObsPy.stream.select` method.
    channel_counts : dict of int
        Number of channels to be used to calculate the onset function for each phase.
        Keys are phases.
    sta_lta_windows : dict of [float, float]
        Short-term average (STA) and Long-term average (LTA) window lengths - keys are
        phases. [STA, LTA] (both in seconds)
    all_channels : bool
        If True, only calculate an onset function when all requested channels meet the
        availability criteria. Otherwise, if at least one channel is available (e.g.
        just the N component for the S phase) the onset function will be calculated from
        that/those.
    allow_gaps : bool
        If True, allow gappy data to be used to calculate the onset function. Gappy data
        will be detrended, tapered and filtered, then gaps padded with zeros. This
        should help mitigate the expected spikes as data goes on- and off-line, but will
        not eliminate it. Onset functions for periods with no data will be filled with
        ~ zeros (smallest possible float, to avoid divide by zero errors). NOTE: This
        feature is experimental and still under development.
    full_timespan : bool
        If False, allow data which doesn't cover the full timespan requested to be used
        for onset function calculation. This is a subtly different test to `allow_gaps`;
        data must be continuous within the timespan, but may not span the whole period.
        Data will be treated as described in `allow_gaps`. NOTE: This feature is
        experimental and still under development.
    position : str, optional
        Compute centred STA/LTA (STA window is preceded by LTA window; value is assigned
        to end of LTA window / start of STA window) or classic STA/LTA (STA window is
        within LTA window; value is assigned to end of STA & LTA windows).
        Default: "classic".

        Centred gives less phase-shifted (late) onset function, and is closer to a
        Gaussian approximation, but is far more sensitive to data with sharp offsets due
        to instrument failures. We recommend using classic for detect() and centred for
        locate() if your data quality allows it. This is the default behaviour; override
        by setting this variable.
    sampling_rate : int
        Desired sampling rate for input data, in Hz; sampling rate at which the onset
        functions will be computed.

    Methods
    -------
    calculate_onsets
        Generate onset functions that represent seismic phase arrivals.
    gaussian_halfwidth
        Phase-appropriate Gaussian half-width estimate based on the short-term average
        window length.

    """

    def __init__(self, **kwargs):
        """Instantiate the STALTAOnset object."""

        super().__init__(**kwargs)

        self.position = kwargs.get("position", "classic")
        self.phases = kwargs.get("phases", ["P", "S"])
        self.bandpass_filters = kwargs.get(
            "bandpass_filters", {"P": [2.0, 16.0, 2], "S": [2.0, 16.0, 2]}
        )
        self.sta_lta_windows = kwargs.get(
            "sta_lta_windows", {"P": [0.2, 1.0], "S": [0.2, 1.0]}
        )
        self.channel_maps = kwargs.get("channel_maps", {"P": "*Z", "S": "*[N,E,1,2]"})
        self.channel_counts = kwargs.get("channel_counts", {"P": 1, "S": 2})
        self.all_channels = kwargs.get("all_channels", False)
        self.allow_gaps = kwargs.get("allow_gaps", False)
        self.full_timespan = kwargs.get("full_timespan", True)

        # --- Deprecated ---
        self.onset_centred = kwargs.get("onset_centred")
        self.p_bp_filter = kwargs.get("p_bp_filter")
        self.s_bp_filter = kwargs.get("s_bp_filter")
        self.p_onset_win = kwargs.get("p_onset_win")
        self.s_onset_win = kwargs.get("s_onset_win")

    def __str__(self):
        """Return short summary string of the Onset object."""

        out = (
            f"\tOnset parameters - using the {self.position} STA/LTA onset"
            f"\n\t\tOnset function sampling rate = {self.sampling_rate} Hz"
            f"\n\t\tPhase(s) = {self.phases}\n"
        )
        for phase, filt in self.bandpass_filters.items():
            out += f"\n\t\t{phase} bandpass filter  = {filt} (Hz, Hz, -)"
        out += "\n"
        for phase, windows in self.sta_lta_windows.items():
            out += f"\n\t\t{phase} onset [STA, LTA] = {windows} (s, s)"
        out += "\n"

        return out

    def calculate_onsets(self, data, log=True, timespan=None):
        """
        Calculate onset functions for the requested stations and phases.

        Returns a stacked array of onset functions for the requested phases, and an
        :class:`~quakemigrate.signal.onsets.base.OnsetData` object containing all
        outputs from the onset function calculation: a dict of the onset functions, a
        Stream containing the pre-processed input waveforms, and a dict of availability
        info describing which of the requested onset functions could be calculated
        (depending on data availability and data quality checks).

        Parameters
        ----------
        data : :class:`~quakemigrate.io.data.WaveformData` object
            Light class encapsulating data returned by an archive query.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.
        timespan : float or None, optional
            If the timespan for which the onsets are being generated is provided, this
            will be used to calculate the tapered window of data at the start and end of
            the onset function which should be disregarded. This is necessary to
            accurately set the pick threshold in GaussianPicker, for example.

        Returns
        -------
        onsets : `numpy.ndarray` of float
            Stacked onset functions served up for migration, shape(nonsets, nsamples).
        onset_data : :class:`~quakemigrate.signal.onsets.base.OnsetData` object
            Light class encapsulating data generated during onset calculation.

        """

        onsets = []
        onsets_dict = {}
        filtered_waveforms = Stream()
        availability = {}

        # Loop through phases, pre-process data, and calculate onsets.
        for phase in self.phases:
            # Select traces based on channel map for this phase
            phase_waveforms = data.waveforms.select(channel=self.channel_maps[phase])

            # Convert sta window, lta window lengths from seconds to samples.
            stw, ltw = self.sta_lta_windows[phase]
            stw = util.time2sample(stw, self.sampling_rate) + 1
            ltw = util.time2sample(ltw, self.sampling_rate) + 1

            # Pre-process the data. The ObsPy functions operate by trace, so
            # will not break on gappy data (we haven't checked availability
            # yet)
            filtered_phase_waveforms = pre_process(
                phase_waveforms,
                self.sampling_rate,
                data.resample,
                data.upfactor,
                self.bandpass_filters[phase],
                data.starttime,
                data.endtime,
            )

            # Loop through stations, check data availability for this phase,
            # and store this info, filtered waveforms and calculated onsets
            for station in data.stations:
                waveforms = filtered_phase_waveforms.select(station=station)

                available, av_dict = data.check_availability(
                    waveforms,
                    all_channels=self.all_channels,
                    n_channels=self.channel_counts[phase],
                    allow_gaps=self.allow_gaps,
                    full_timespan=self.full_timespan,
                    check_sampling_rate=True,
                    sampling_rate=self.sampling_rate,
                )
                availability[f"{station}_{phase}"] = available

                # If no data available, skip
                if available == 0:
                    logging.info(f"\t\tNo {phase} onset for {station}.")
                    continue

                # Check that all channels met the availability critera. If
                # not, remove this channel from the stream.
                for key, available in av_dict.items():
                    if available == 0:
                        to_remove = waveforms.select(id=key)
                        [waveforms.remove(tr) for tr in to_remove]

                # Pad with tiny floats so onset will be the correct length.
                # Note: this will only have an effect if allow_gaps=True or
                # full_timespan=False. Otherwise, there will be no gaps to pad.
                if self.allow_gaps or not self.full_timespan:
                    # Square root to avoid floating point errors when value
                    # is squared to compute the energy trace
                    tiny = np.sqrt(np.finfo(float).tiny)
                    # Apply another taper to remove transients from filtering -
                    # this is within the pre- and post-pad for continuous data
                    waveforms.taper(type="cosine", max_percentage=0.05)
                    # Fill gaps
                    waveforms.merge(method=1, fill_value=tiny)
                    # Pad start/end; delta of +/-0.00001 is to avoid
                    # occasional obspy weirdness. `nearest_sample` is
                    # appropriate as data is at uniform sampling rate with
                    # off-sample data corrected by util.shift_to_sample()
                    waveforms.trim(
                        starttime=data.starttime - 0.00001,
                        endtime=data.endtime + 0.00001,
                        pad=True,
                        fill_value=tiny,
                        nearest_sample=False,
                    )

                # Calculate onset and add to WaveForm data object; add filtered
                # waveforms that have passed the availability check to
                # WaveformData.filtered_waveforms
                onsets_dict.setdefault(station, {}).update(
                    {phase: self._onset(waveforms, stw, ltw, log, timespan)}
                )
                onsets.append(onsets_dict[station][phase])
                filtered_waveforms += waveforms

        logging.debug(filtered_waveforms.__str__(extended=True))

        if sum(availability.values()) == 0:
            raise util.DataAvailabilityException

        onsets = np.stack(onsets, axis=0)
        onset_data = OnsetData(
            onsets_dict,
            self.phases,
            self.channel_maps,
            filtered_waveforms,
            availability,
            data.starttime,
            data.endtime,
            self.sampling_rate,
        )

        return onsets, onset_data

    def _onset(self, stream, stw, ltw, log, timespan):
        """
        Generates an onset (characteristic) function. If there are multiple components,
        these are combined as the root-mean-square of the onset functions AFTER taking a
        log (if requested).

        Parameters
        ----------
        stream : `obspy.Stream` object
            Stream containing the pre-processed data from which to calculate the onset
            function.
        stw : int
            Number of samples in the short-term window.
        ltw : int
            Number of samples in the long-term window.
        log : bool
            Calculate log(onset) if True, otherwise calculate the raw onset.
        timespan : float or None
            If a timespan is provided it will be used to calculate the tapered window of
            data at the start and end of the onset function which should be disregarded.

        Returns
        -------
        onset : `numpy.ndarray` of float
            STA/LTA onset function.

        """

        if self.position == "centred":
            onsets = [sta_lta_centred(tr.data, stw, ltw) for tr in stream]
        elif self.position == "classic":
            onsets = [classic_sta_lta(tr.data, stw, ltw) for tr in stream]
        onsets = np.array(onsets)

        if timespan:
            onsets = self._trim_taper_pad(onsets, stw, ltw, timespan)

        np.clip(1 + onsets, 0.8, np.inf, onsets)
        if log:
            np.log(onsets, onsets)

        onset = np.sqrt(np.sum([onset**2 for onset in onsets], axis=0) / len(onsets))

        return onset

    def _trim_taper_pad(self, onsets, stw, ltw, timespan):
        """
        Set the value of the tapered windows at the start and end of the onset function
        (plus one long-term window and one short-term window, respectively) to 0.

        Parameters
        ----------
        onsets : `numpy.ndarray` of float
            STA/LTA onset function.
        stw : int
            Number of samples in the short-term window.
        ltw : int
            Number of samples in the long-term window.
        timespan : float
            Used to calculate the tapered window of data at the start and end of the
            onset function which should be disregarded.

        Returns
        -------
        onsets : `numpy.ndarray` of float
            STA/LTA onset function, with the value in the tapered regions of data set to
            0.

        """

        # Calculate duration of taper pre- and post-pad and convert to samples
        pre_pad, _ = self.pad(timespan)
        # Taper pre- and post-pad are identical - just calculate one
        taper_pad = util.time2sample(pre_pad - self.pre_pad, self.sampling_rate)

        for onset in onsets:
            onset[: (taper_pad + ltw - 1)] = 0
            onset[-(stw + taper_pad) :] = 0

        return onsets

    def gaussian_halfwidth(self, phase):
        """
        Return the phase-appropriate Gaussian half-width estimate based on the
        short-term average window length.

        Parameters
        ----------
        phase : {'P', 'S'}
            Seismic phase for which to serve the estimate.

        """

        return self.sta_lta_windows[phase][0] * self.sampling_rate / 2

    @property
    def pre_pad(self):
        """Pre-pad is determined as a function of the onset windows"""
        windows = self.sta_lta_windows
        pre_pad = max([windows[key][1] for key in windows.keys()]) + 3 * max(
            [windows[key][0] for key in windows.keys()]
        )

        return pre_pad

    @pre_pad.setter
    def pre_pad(self, value):
        """Setter for pre-pad"""

        self._pre_pad = value

    @property
    def post_pad(self):
        """
        Post-pad is determined as a function of the max traveltime in the grid and the
        onset windows

        """

        return self._post_pad

    @post_pad.setter
    def post_pad(self, ttmax):
        """
        Define post-pad as a function of the maximum travel-time between a station and a
        grid point plus the LTA (in case onset_centred is True)

        """
        windows = self.sta_lta_windows
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
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, change:\n"
            "\t'onset_centred' -> 'position'"
        )
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
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, refer to the documentation."
        )

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
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, refer to the documentation."
        )

        self.bandpass_filters["S"] = value

    @property
    def p_onset_win(self):
        """Handle deprecated p_onset_win kwarg / attribute"""
        return self.sta_lta_windows["P"]

    @p_onset_win.setter
    def p_onset_win(self, value):
        """
        Handle deprecated p_onset_win kwarg / attribute and print warning.

        """
        if value is None:
            return
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, refer to the documentation."
        )

        self.sta_lta_windows["P"] = value

    @property
    def s_onset_win(self):
        """Handle deprecated s_onset_win kwarg / attribute"""
        return self.sta_lta_windows["S"]

    @s_onset_win.setter
    def s_onset_win(self, value):
        """
        Handle deprecated s_onset_win kwarg / attribute and print warning.

        """
        if value is None:
            return
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, refer to the documentation."
        )

        self.sta_lta_windows["S"] = value


class CentredSTALTAOnset(STALTAOnset):
    """
    QuakeMigrate default onset function class - uses a centred STA/LTA onset.

    NOTE: THIS CLASS HAS BEEN DEPRECATED AND WILL BE REMOVED IN A FUTURE UPDATE

    """

    def __init__(self, **kwargs):
        """Instantiate CentredSTALTAOnset object."""

        super().__init__(**kwargs)

        print(
            "FutureWarning: This class has been deprecated - continuing.\n"
            "To remove this message:\n"
            "\tCentredSTALTAOnset -> STALTAOnset\n"
            "\tAnd add keyword argument 'position=centred'\n"
        )
        self.position = "centred"


class ClassicSTALTAOnset(STALTAOnset):
    """
    QuakeMigrate default onset function class - uses a classic STA/LTA onset.

    NOTE: THIS CLASS HAS BEEN DEPRECATED AND WILL BE REMOVED IN A FUTURE UPDATE

    """

    def __init__(self, **kwargs):
        """Instantiate ClassicSTALTAOnset object."""

        super().__init__(**kwargs)

        print(
            "FutureWarning: This class has been deprecated - continuing.\n"
            "To remove this message:\n"
            "\tClassicSTALTAOnset -> STALTAOnset\n"
            "\tAnd add keyword argument 'position=classic'\n"
        )
        self.position = "classic"
