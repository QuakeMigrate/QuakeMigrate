"""
The default onset function class - performs some pre-processing on raw seismic data and
calculates STA/LTA onset (characteristic) function.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from typing import Literal, TYPE_CHECKING

import numpy as np
from obspy import Stream
from scipy.signal import hilbert

import quakemigrate.util as util
from quakemigrate.core import overlapping_sta_lta, centred_sta_lta
from quakemigrate.exceptions import AllDataRejected, NyquistException
from quakemigrate.plugins.onsets.base import Onset, OnsetData


if TYPE_CHECKING:
    from obspy import UTCDateTime

    from quakemigrate.io.data import WaveformData


def centred_sta_lta_py(signal: np.ndarray, nsta: int, nlta: int) -> np.ndarray:
    """
    Calculates the ratio of the average of the signal in a short-term (signal) window to
    a preceding long-term (noise) window. STA/LTA value is assigned to the end of the
    LTA / one sample before the start of the STA.
    NOTE: signal must be non-negative.

    Parameters
    ----------
    signal:
        Transformed non-negative seismic waveform.
    nsta:
        Number of samples in short-term window.
    nlta:
        Number of samples in long-term window.

    Returns
    -------
    sta/lta:
        Short-term average / long-term average ratio of the signal amplitude, computed
        in adjacent STA/LTA windows.

    """

    # Cumulative sum to calculate moving average
    sta = np.cumsum(signal)

    # Convert to float
    sta = np.require(sta, dtype=float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta[nsta:-nsta] = sta[nsta * 2 :]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad with ones (= null result)
    sta[: nlta - 1] = 1.0
    lta[: nlta - 1] = 1.0
    sta[-nsta:] = 1.0
    lta[-nsta:] = 1.0

    # Avoid division by zero by setting zero values to tiny float, giving an
    # STA/LTA of 1 (= null result)
    dtiny = np.finfo(float).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = dtiny

    return sta / lta


def overlapping_sta_lta_py(signal: np.ndarray, nsta: int, nlta: int) -> np.ndarray:
    """
    Computes the standard STA/LTA from a given input array `signal`. The length of the
    STA window is given by nsta (in samples), nlta is the length of the LTA window (in
    samples). STA window fully overlaps with the LTA window, and is positioned to the
    "right" i.e. the end of both of the windows is at the latest point in time; this is
    where the STA / LTA value is assigned.
    NOTE: signal must be non-negative.

    Parameters
    ----------
    signal:
        Transformed non-negative seismic waveform.
    nsta:
        Length of short time average window in samples.
    nlta:
        Length of long time average window in samples.

    Returns
    -------
    sta/lta:
        Short-term average / long-term average ratio of the signal amplitude, computed
        in overlapping STA/LTA windows.

    """

    # Cumulative sum to calculate moving average
    sta = np.cumsum(signal)

    # Convert to float
    sta = np.require(sta, dtype=float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad with ones (= null result)
    sta[: nlta - 1] = 1.0
    lta[: nlta - 1] = 1.0

    # Avoid division by zero by setting zero values to tiny float, giving an
    # STA/LTA of 1 (= null result)
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = dtiny

    return sta / lta


def pre_process(
    stream: Stream,
    sampling_rate: int,
    resample: bool,
    upfactor: int,
    filter_: list[float],
    starttime: UTCDateTime,
    endtime: UTCDateTime,
) -> Stream:
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
    stream:
        Waveform data to be pre-processed.
    sampling_rate:
        Desired sampling rate for data to be used to calculate onset. This will be
        achieved by resampling the raw waveform data. By default, only decimation will
        be applied, but data can also be upsampled if specified by the user when
        creating the :class:`~quakemigrate.io.data.Archive` object.
    resample:
        If true, perform resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
    upfactor:
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g., 40Hz -> 50Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    filter_:
        Filter specifications, as [lowcut (Hz), highcut (Hz), order]. NOTE - two-pass
        filter effectively doubles the number of corners (order).
    starttime:
        Timestamp of first sample in waveform data.
    endtime:
        Timestamp of last sample in waveform data.

    Returns
    -------
    filtered_waveforms:
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
        raise NyquistException(highcut, 0.5 * sampling_rate, "")

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
    STA/LTA onset plugin for QuakeMigrate.

    This onset function preprocesses raw seismic waveform data, including re-sampling if
    necessary to reach the specified uniform sampling rate, checks station and phase
    availability against a user-specified set of data quality criteria, transforms
    waveform amplitudes into a non-negative characteristic signal, and calculates
    phase-specific onset functions using short-term average / long-term average ratios.

    The class is QuakeMigrate's canonical built-in onset implementation and demonstrates
    the expected plugin contract: validated configuration, consistent station/phase
    bookkeeping, robust availability handling, and standard
    :class:`~quakemigrate.plugins.onsets.base.OnsetData` output.

    Attributes
    ----------
    sampling_rate:
        Desired sampling rate for input data, in Hz; sampling rate at which the onset
        functions will be computed.
    position:
        Compute centred STA/LTA (STA window is preceded by LTA window; value is assigned
        to end of LTA window / start of STA window) or classic STA/LTA (STA window is
        within LTA window; value is assigned to end of STA & LTA windows).

        Centred gives less phase-shifted (late) onset function, and is closer to a
        Gaussian approximation, but is far more sensitive to data with sharp offsets due
        to instrument failures. We recommend using classic for detect() and centred for
        locate() if your data quality allows it. This is the default behaviour; override
        by setting this variable.
    use_python_backend:
        Toggle to use Python implementations of onset functions.
    signal_transform:
        Transformation to apply to the signal before taking the STA/LTA, to ensure the
        signal is always positive: energy (signal^2), absolute value, envelope (absolute
        value of the analytic signal), or envelope^2 (analytic, and arguably more
        correct, measure of the energy of the signal).
    min_onset_value:
        Minimum value at which to clip the onset function. This is the equivalent to
        setting a minimum SNR filter for which observations to include. The appropriate
        value will depend on the signal and noise characteristics, and the
        `signal_transform` selected.
        NOTE: must be greater than 0.01
    phases:
        Which phases to calculate onset functions for. This will determine which phases
        are used for migration/picking. The selected phases must be present in the
        traveltime lookup table to be used for these purposes.
    bandpass_filters:
        Butterworth bandpass filter specification - keys are phases.
        [lowpass (Hz), highpass (Hz), corners*]
        *NOTE: two-pass filter effectively doubles the number of corners.
    sta_lta_windows:
        Short-term average (STA) and Long-term average (LTA) window lengths - keys are
        phases. [STA, LTA] (both in seconds)
    channel_maps:
        Data component maps - keys are phases. These are passed into the
        :meth:`ObsPy.stream.select` method.
    channel_counts:
        Number of channels to be used to calculate the onset function for each phase.
        Keys are phases.
    all_channels:
        If True, only calculate an onset function when all requested channels meet the
        availability criteria. Otherwise, if at least one channel is available (e.g.,
        just the N component for the S phase) the onset function will be calculated from
        that/those.
    allow_gaps:
        If True, allow gappy data to be used to calculate the onset function. Gappy data
        will be detrended, tapered and filtered, then gaps padded with zeros. This
        should help mitigate the expected spikes as data goes on- and off-line, but will
        not eliminate it. Onset functions for periods with no data will be filled with
        ~ zeros (smallest possible float, to avoid divide by zero errors).
        NOTE: This feature is experimental and still under development.
    full_timespan:
        If False, allow data which doesn't cover the full timespan requested to be used
        for onset function calculation. This is a subtly different test to `allow_gaps`;
        data must be continuous within the timespan, but may not span the whole period.
        Data will be treated as described in `allow_gaps`. NOTE: This feature is
        experimental and still under development.

    Raises
    ------
    ValueError
        If the minimum onset value is less than 0.01.

    """

    def __init__(
        self,
        *,
        sampling_rate: int,
        position: Literal["classic", "centred"] = "classic",
        use_python_backend: bool = False,
        signal_transform: Literal["energy", "abs", "env", "env_squared"] = "energy",
        min_onset_value: float = 0.4,
        phases: list[str] | None = None,
        bandpass_filters: dict[str, list[float | int]] | None = None,
        sta_lta_windows: dict[str, list[float]] | None = None,
        channel_maps: dict[str, str] | None = None,
        channel_counts: dict[str, int] | None = None,
        all_channels: bool = False,
        allow_gaps: bool = False,
        full_timespan: bool = True,
    ) -> None:
        """Instantiate the STALTAOnset object."""

        super().__init__(sampling_rate=sampling_rate)

        # --- General parameters ---
        self.position = position
        self.use_python_backend = use_python_backend
        self.signal_transform = signal_transform

        if min_onset_value < 0.01:
            raise ValueError("The `min_onset_value` must be greater than 0.01")
        self.min_onset_value = min_onset_value

        # --- Phase-specific parameters ---
        self.phases = ["P", "S"] if phases is None else phases
        self.bandpass_filters = (
            {"P": [2.0, 16.0, 2], "S": [2.0, 16.0, 2]}
            if bandpass_filters is None
            else bandpass_filters
        )
        self.sta_lta_windows = (
            {"P": [0.2, 1.0], "S": [0.2, 1.0]}
            if sta_lta_windows is None
            else sta_lta_windows
        )
        self.channel_maps = (
            {"P": "*Z", "S": "*[N,E,1,2]"} if channel_maps is None else channel_maps
        )
        self.channel_counts = (
            {"P": 1, "S": 2} if channel_counts is None else channel_counts
        )

        self.all_channels = all_channels
        self.allow_gaps = allow_gaps
        self.full_timespan = full_timespan

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate phase-specific STALTA configuration."""

        for phase in self.phases:
            if phase not in self.bandpass_filters:
                raise ValueError(f"No bandpass filter specified for phase {phase}")

            if phase not in self.sta_lta_windows:
                raise ValueError(f"No STA/LTA window specified for phase {phase}")

            if phase not in self.channel_maps:
                raise ValueError(f"No channel map specified for phase {phase}")

            if phase not in self.channel_counts:
                raise ValueError(f"No channel count specified for phase {phase}")

    def __str__(self) -> str:
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

    def calculate_onsets(
        self, data: WaveformData, timespan: float | None = None
    ) -> tuple[np.ndarray, OnsetData]:
        """
        Calculate onset functions for the requested stations and phases.

        Returns a stacked array of onset functions for the requested phases, and an
        :class:`~quakemigrate.plugins.onsets.base.OnsetData` object containing all
        outputs from the onset function calculation: a dict of the onset functions, a
        Stream containing the pre-processed input waveforms, and a dict of availability
        info describing which of the requested onset functions could be calculated
        (depending on data availability and data quality checks).

        Parameters
        ----------
        data:
            Waveform data returned by an archive query.
        timespan:
            If the timespan for which the onsets are being generated is provided, this
            will be used to calculate the tapered window of data at the start and end of
            the onset function which should be disregarded. This is necessary to
            accurately set the pick threshold in GaussianPicker, for example.

        Returns
        -------
        onsets:
            Stacked onset functions served up for migration, shape(nonsets, nsamples).
        onset_data:
            Light class encapsulating data generated during onset calculation.

        Raises
        ------
        AllDataRejected
            If no data passes the specified criteria.

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
            for station in [s for s in data.stations if not s.read_only]:
                waveforms = filtered_phase_waveforms.select(station=station.station)

                available, av_dict = data.check_availability(
                    waveforms,
                    all_channels=self.all_channels,
                    n_channels=self.channel_counts[phase],
                    allow_gaps=self.allow_gaps,
                    full_timespan=self.full_timespan,
                    check_sampling_rate=True,
                    sampling_rate=self.sampling_rate,
                )
                availability[f"{station.id}_{phase}"] = available

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
                onsets_dict.setdefault(station.id, {}).update(
                    {phase: self._onset(waveforms, stw, ltw, timespan)}
                )
                onsets.append(onsets_dict[station.id][phase])
                filtered_waveforms += waveforms

        logging.debug(filtered_waveforms.__str__(extended=True))

        if sum(availability.values()) == 0:
            raise AllDataRejected()

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

    def _onset(
        self, stream: Stream, stw: int, ltw: int, timespan: float | None
    ) -> np.ndarray:
        """
        Generates an onset (characteristic) function. If there are multiple components,
        these are combined as the root-mean-square of the onset functions.

        Parameters
        ----------
        stream:
            Stream containing the pre-processed data from which to calculate the onset
            function.
        stw:
            Number of samples in the short-term window.
        ltw:
            Number of samples in the long-term window.
        timespan:
            If a timespan is provided it will be used to calculate the tapered window of
            data at the start and end of the onset function which should be disregarded.

        Returns
        -------
        onset:
            STA/LTA onset function.

        """

        if self.signal_transform == "energy":
            transformed_waveforms = [tr.data**2 for tr in stream]
        elif self.signal_transform == "abs":
            transformed_waveforms = [np.abs(tr.data) for tr in stream]
        elif self.signal_transform == "env":
            transformed_waveforms = [np.abs(hilbert(tr.data)) for tr in stream]
        elif self.signal_transform == "env_squared":
            transformed_waveforms = [np.abs(hilbert(tr.data)) ** 2 for tr in stream]

        if self.position == "centred":
            if self.use_python_backend:
                onset_fn = centred_sta_lta_py
            else:
                onset_fn = centred_sta_lta
        elif self.position == "classic":
            if self.use_python_backend:
                onset_fn = overlapping_sta_lta_py
            else:
                onset_fn = overlapping_sta_lta

        onsets = np.array(
            [onset_fn(waveform, stw, ltw) for waveform in transformed_waveforms]
        )

        if timespan:
            onsets = self._trim_taper_pad(onsets, stw, ltw, timespan)

        # Combine onsets when using multiple components
        onset = np.sqrt(np.sum([onset**2 for onset in onsets], axis=0) / len(onsets))

        onset = np.clip(onset, self.min_onset_value, np.inf)

        return onset

    def _trim_taper_pad(
        self, onsets: np.ndarray, stw: int, ltw: int, timespan: float
    ) -> np.ndarray:
        """
        Set the value of the tapered windows at the start and end of the onset function
        (plus one long-term window and one short-term window, respectively) to 1.

        Parameters
        ----------
        onsets:
            STA/LTA onset function.
        stw:
            Number of samples in the short-term window.
        ltw:
            Number of samples in the long-term window.
        timespan:
            Used to calculate the tapered window of data at the start and end of the
            onset function which should be disregarded.

        Returns
        -------
        onsets:
            STA/LTA onset function, with the value in the tapered regions of data set to
            1.

        """

        # Calculate duration of taper pre- and post-pad and convert to samples
        pre_pad, _ = self.pad(timespan)
        # Taper pre- and post-pad are identical - just calculate one
        taper_pad = util.time2sample(pre_pad - self.pre_pad, self.sampling_rate)

        for onset in onsets:
            onset[: (taper_pad + ltw - 1)] = 1.0
            onset[-(stw + taper_pad) :] = 1.0

        return onsets

    def gaussian_halfwidth(self, phase: str) -> float:
        """
        Return the phase-appropriate Gaussian half-width estimate based on the
        short-term average window length.

        Parameters
        ----------
        phase:
            Seismic phase for which to serve the estimate.

        Returns
        -------
        halfwidth:
            The Gaussian halfwidth estimate based on the STA window length.

        """

        return self.sta_lta_windows[phase][0] * self.sampling_rate / 2

    @property
    def pre_pad(self) -> float:
        """Pre-pad is determined as a function of the onset windows"""
        windows = self.sta_lta_windows
        pre_pad = max([windows[key][1] for key in windows.keys()]) + 3 * max(
            [windows[key][0] for key in windows.keys()]
        )

        return pre_pad

    @pre_pad.setter
    def pre_pad(self, value: float) -> None:
        """Setter for pre-pad"""

        self._pre_pad = value

    @property
    def post_pad(self) -> float:
        """
        Post-pad is determined as a function of the max traveltime in the grid and the
        onset windows

        """

        return self._post_pad

    @post_pad.setter
    def post_pad(self, ttmax: float) -> None:
        """
        Define post-pad as a function of the maximum travel-time between a station and a
        grid point plus the LTA (in case onset_centred is True)

        """
        windows = self.sta_lta_windows
        lta_max = max([windows[key][1] for key in windows.keys()])
        self._post_pad = np.ceil(ttmax + 2 * lta_max)
