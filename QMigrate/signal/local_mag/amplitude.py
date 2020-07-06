# -*- coding: utf-8 -*-
"""
Module containing methods to measure Wood-Anderson corrected waveform
amplitudes to be used for local magnitude calculation.

"""

import logging

import numpy as np
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
import pandas as pd
from scipy.signal import find_peaks, iirfilter, sosfreqz

import QMigrate.util as util


class Amplitude:
    """
    Part of the QuakeMigrate LocalMag class; measures Wood-Anderson corrected
    waveform amplitudes to be used for local magnitude calculation.

    Simulates the Wood-Anderson waveforms using a user-supplied set of response
    removal parameters, then measures the maximum peak-to-trough amplitude in
    time windows around the P and S phase arrivals. These windows are
    calculated from the phase pick times from the autopicker, if available, or
    from the modelled pick times. The length of the S-wave signal window is
    calculated according to a user-specified `signal_window` parameter.

    The user may optionally specify a filter to apply to the waveforms before
    amplitudes are measured, in order (for example) to reduce the impact of
    high-amplitude noise associated with the oceanic microseisms on the
    measurement of low-amplitude wavetrains associated with microseismic
    events. Note this will generally result in an underestimate of the true
    earthquake waveform amplitude, even when the filter gain is corrected for.

    A measurement of the signal amplitude in a window preceding the P-wave
    arrival is used to characterise the "noise" amplitude. This can be used
    to filter out null observations, and to provide an estimate of the
    uncertainty on the max amplitude measurements contributed by extraneous
    noise.

    Attributes
    ----------
    pre_filt : tuple of floats
        Pre-filter to apply during the instrument response removal. E.g.
        (0.03, 0.05, 30., 35.) - all in Hz. (Default None)
    water_level : float
        Water level to use in instrument response removal. (Default 60.)
    signal_window : float
        Length of S-wave signal window, in addition to the time window
        associated with the marginal_window and traveltime uncertainty.
        (Default 0 s)
    noise_window : float
        Length of the time window before the P-wave signal window in which
        to measure the noise amplitude. (Default 5 s)
    noise_measure : {"RMS", "STD"}
        Method by which to measure the noise amplitude; root-mean-quare or
        standard deviation of the signal. (Default "RMS")
    loc_method : {"spline", "gaussian", "covariance"}
        Which event location estimate to use. (Default "spline")
    remove_full_response : bool
        Whether to remove the full response (including the effect of
        digital FIR filters) or just the instrument transform function (as
        defined by the PolesZeros Response Stage). Significantly slower.
        (Default False)
    highpass_filter : bool
        Whether to apply a high-pass filter before measuring amplitudes.
        (Default False)
    highpass_freq : float
        High-pass filter frequency. Required if highpass_filter is True.
    bandpass_filter : bool
        Whether to apply a band-pass filter before measuring amplitudes.
        (Default False)
    bandpass_lowcut : float
        Band-pass filter low-cut frequency. Required if bandpass_filter is
        True.
    bandpass_highcut : float
        Band-pass filter high-cut frequency. Required if bandpass_filter is
        True.
    filter_corners : int
        number of corners for the chosen filter. (Default 4)
    prominence_multiplier : float
        To set a prominence filter in the peak-finding algorithm.
        (Default 0. = off)
        NOTE: not recommended for use in combination with a filter; filter gain
        corrections can lead to spurious results. Please see the
        `scipy.signal.find_peaks` documentation for further guidance.

    Methods
    -------
    get_amplitudes(event, lut)

    Raises
    ------
    AttributeError
        If both highpass_filter and bandpass_filter are selected, or if the
        user selects to apply a filter but does not provide the relevant
        frequencies.

    """

    def __init__(self, amplitude_params={}):
        """Instantiate the Amplitude object."""

        # Amplitude measurement parameters
        if "signal_window" not in amplitude_params.keys():
            msg = ("Warning: 'signal_window' not specified. Set to default: 0")
            logging.warning(msg)
        self.signal_window = amplitude_params.get("signal_window", 0.)

        self.noise_window = amplitude_params.get("noise_window", 5.)
        self.noise_measure = amplitude_params.get("noise_measure", "RMS")

        self.prominence_multiplier = \
            amplitude_params.get("prominence_multiplier", 0.)
        self.loc_method = amplitude_params.get("loc_method", "spline")

        # Pre-processing parameters
        self.highpass_filter = amplitude_params.get("highpass_filter", False)
        if self.highpass_filter:
            try:
                self.highpass_freq = amplitude_params["highpass_freq"]
            except KeyError as e:
                msg = f"Highpass filter frequency not specified! {e}"
                raise AttributeError(msg)

        self.bandpass_filter = amplitude_params.get("bandpass_filter", False)
        if self.bandpass_filter:
            try:
                self.bandpass_lowcut = amplitude_params.get("bandpass_lowcut")
                self.bandpass_highcut = amplitude_params.get("bandpass_highcut")
            except KeyError as e:
                msg = f"Bandpass filter frequencies not specified! {e}"
                raise AttributeError(msg)
        self.filter_corners = amplitude_params.get("filter_corners", 4)

        if self.highpass_filter and self.bandpass_filter:
            msg = ("Both bandpass filter *and* highpass filter selected! "
                   "Please choose one or the other.")
            raise AttributeError(msg)

        # Response removal parameters
        if "water_level" not in amplitude_params.keys():
            msg = ("Warning: 'water level' for instrument correction not "
                   "specified. Set to default: 60")
            logging.warning(msg)
        self.water_level = amplitude_params.get("water_level", 60.)
        self.pre_filt = amplitude_params.get("pre_filt")
        self.remove_full_response = \
            amplitude_params.get("remove_full_response", False)

    def __str__(self):
        """Return short summary string of the Amplitude object."""

        out = ("\t    Amplitude parameters:\n"
               f"\t\tSignal window    = {self.signal_window} s\n"
               f"\t\tNoise window     = {self.noise_window} s\n"
               f"\t\tNoise measure    = {self.noise_measure}\n"
               f"\t\tLocation used    = {self.loc_method}\n")
        if self.prominence_multiplier != 0.:
            out += f"\t\tProminence multiplier = {self.prominence_multiplier}"
            out += "\n"
        if self.highpass_filter:
            out += ("\t\tHighpass filter: \n"
                    f"\t\t    Filter frequency = {self.highpass_freq} Hz\n"
                    f"\t\t    Filter corners   = {self.filter_corners}\n")
        elif self.bandpass_filter:
            out += ("\t\tBandpass filter: \n"
                    f"\t\t    Lowcut frequency  = {self.bandpass_lowcut} Hz\n"
                    f"\t\t    Highcut frequency = {self.bandpass_highcut} Hz\n"
                    f"\t\t    Filter corners    = {self.filter_corners}\n")
        out += ("\t    Response removal parameters:\n"
                f"\t\tWater level  = {self.water_level}\n")
        if self.pre_filt is not None:
            out += f"\t\tPre-filter   = {self.pre_filt} Hz\n"
        out += ("\t\tRemove full response (inc. FIR stages) = "
                f"{self.remove_full_response}\n")

        return out

    @util.timeit
    def get_amplitudes(self, event, lut):
        """
        Measure phase amplitudes for an event.

        Parameters
        ----------
        event : `QMigrate.io.Event` object
            Light class encapsulating waveform data, onset, pick and location
            information for a given event.
        lut : `QMigrate.lut.LUT` object
            Contains the traveltime lookup tables for seismic phases, computed
            for some pre-defined velocity model.

        Returns
        -------
        amplitudes : `pandas.DataFrame` object
            P- and S-wave amplitude measurements for each component of each
            station in the station file.
            Columns:
                epi_dist : float
                    Epicentral distance between the station and the event
                    hypocentre.
                z_dist : float
                    Vertical distance between the station and the event
                    hypocentre.
                P_amp : float
                    Half maximum peak-to-trough amplitude in the P signal
                    window. In *millimetres*.
                P_freq : float
                    Approximate frequency of the maximum amplitude P-wave
                    signal. Calculated from the peak-to-trough time of the max
                    peak-to-trough amplitude.
                P_time : `obspy.UTCDateTime` object
                    Approximate time of amplitude observation (halfway between
                    peak and trough times).
                S_amp : float
                    As for P, but in the S wave signal window.
                S_freq : float
                    As for P.
                S_time : `obspy.UTCDateTime` object
                    As for P.
                Noise_amp : float
                    An estimate of the signal amplitude in the noise window. In
                    millimetres.
                is_picked : bool
                    Whether at least one of the phase arrivals was picked by
                    the autopicker.
            Index = Trace ID (see `obspy.Trace` object property 'id')

        """

        # Get event hypocentre location
        ev_loc = event.get_hypocentre(self.loc_method)

        # Get start of earliest possible noise window and end of latest
        # possible signal window
        max_tt = lut.max_traveltime
        tr_start = event.otime - event.marginal_window - self.noise_window
        tr_end = event.otime + event.marginal_window + \
            (1. + lut.fraction_tt) * max_tt + self.signal_window

        # Get traveltimes for all stations and phases: much quicker than
        # doing this multiple times in the loop
        event_ijk = lut.index2coord(ev_loc, inverse=True)[0]
        p_ttimes = lut.traveltime_to("P", event_ijk)
        s_ttimes = lut.traveltime_to("S", event_ijk)

        # Initialise amplitudes DataFrame
        amplitudes = pd.DataFrame(columns=["id", "epi_dist", "z_dist",
                                           "P_amp", "P_freq", "P_time",
                                           "S_amp", "S_freq", "S_time",
                                           "Noise_amp", "is_picked"])

        # Loop through stations, calculating amplitude info
        for i, station_data in lut.station_data.iterrows():
            station = station_data["Name"]

            epi_dist, z_dist = self._get_distances(ev_loc, station_data,
                                                   lut.unit_conversion_factor)

            # Columns: tr_id, epicentral distance, vertical distance, P_amp,
            #          P_freq, P_time, S_amp, S_freq, S_time, Noise_amp,
            #          picked
            amps_template = ["", epi_dist, z_dist, np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, np.nan, False]

            # Read in raw waveforms
            st = event.data.raw_waveforms.select(station=station)

            for j, comp in enumerate(["E", "N", "Z"]): # NOTE: Will not work with 1, 2 (etc.)
                amps = amps_template.copy()
                tr = st.select(component=comp)
                # if trace is empty (no traces) or there is more than 1 (gaps),
                # or data isn't continuous within the time window where data is
                # needed to make the amplitude measurements, skip
                if bool(tr) and len(tr) == 1 and \
                    tr[0].stats.starttime < (tr_start + tr[0].stats.delta) \
                    and tr[0].stats.endtime > (tr_end - tr[0].stats.delta):
                    tr = tr[0]
                else:
                    amps[0] = f".{station}..{comp}"
                    amplitudes.loc[i*3+j] = amps
                    continue

                amps[0] = tr.id

                # Do response removal
                try:
                    tr = event.data.get_wa_waveform(tr, self.water_level,
                        self.pre_filt,
                        remove_full_response=self.remove_full_response,
                        velocity=False)
                except (util.ResponseNotFoundError,
                        util.ResponseRemovalError) as e:
                    logging.warning(e)
                    amplitudes.loc[i*3+j] = amps
                    continue

                if self.bandpass_filter or self.highpass_filter:
                    filter_sos = self._filter_trace(tr)
                else:
                    filter_sos = None

                windows, picked = self._get_amplitude_windows(station, i,
                                                              event, p_ttimes,
                                                              s_ttimes,
                                                              lut.fraction_tt)

                amps, filter_gain = self._measure_signal_amps(amps, tr,
                                                              windows,
                                                              filter_sos)

                noise_amp = self._measure_noise_amp(tr, windows, filter_gain,
                                                    method=self.noise_measure)

                # Put in relevant columns; noise amp in *millimetres*
                amps[9:11] = noise_amp, picked

                # 3 rows per station; one for each component
                amplitudes.loc[i*3+j] = amps

        amplitudes = amplitudes.set_index("id")

        return amplitudes

    def _get_distances(self, ev_loc, station_data, unit_conversion_factor):
        """
        Get epicentral and vertical distances between a station and an event
        hypocentre.

        Parameters
        ----------
        ev_loc : array-like
            Event hypocentre location in geographic coordinate system.
        station_data : `pandas.Series` object
            Station information - keys: ["Name", "Latitude", "Longitude",
            "Elevation"].
        unit_conversion_factor : float
            A conversion factor based on the lookup table grid projection, used
            to guarantee the distances returned have units of kilometres.

        Returns
        -------
        epi_dist : float
            Epicentral distance between the station and the event hypocentre.
        z_dist : float
            Vertical distance between the station and the event hypocentre.

        """

        # Get station location
        stla, stlo, stel = station_data[["Latitude",
                                         "Longitude",
                                         "Elevation"]].values

        # Get event location
        evlo, evla, evdp = ev_loc

        # For amplitudes and magnitude calculation, distances must be in km
        km_cf = 1000 / unit_conversion_factor

        # Evaluate epicentral/vertical distances between station and event
        epi_dist = gps2dist_azimuth(evla, evlo, stla, stlo)[0] / km_cf
        z_dist = (evdp - stel) / km_cf

        return epi_dist, z_dist

    def _filter_trace(self, tr):
        """
        Apply a highpass or bandpass filter to the supplied Trace. Filtering is
        applied in-place on the `obspy.Trace` object.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace to be filtered

        Returns
        -------
        filter_sos : ndarray
            Second-order sections representation of the applied filter.

        """

        # Try to apply bandpass filter, unless the specified lowpass frequency
        # is higher than the Nyquist. In this case, apply a highpass.
        if self.bandpass_filter:
            try:
                filter_sos = self._bandpass_filter(tr)
            except util.NyquistException as e:
                logging.warning(f"\t{e}Applying a high-pass filter instead...")
                filter_sos = self._highpass_filter(tr)
        else:
            filter_sos = self._highpass_filter(tr)

        return filter_sos

    def _bandpass_filter(self, tr):
        """
        Apply a bandpass filter to the supplied `obspy.Trace` object; filter
        operation is applied in-place.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace to be filtered.

        Returns
        -------
        filter_sos : ndarray
            Second-order sections representation of the applied filter.

        Raises
        ------
        NyquistException
            If the high-cut filter specified for the bandpass filter is higher
            than the Nyquist frequency of a trace.

        """

        freqmin = self.bandpass_lowcut
        freqmax = self.bandpass_highcut
        corners = self.filter_corners

        # Check specified freqmax is possible for this trace
        f_nyquist = 0.5 * tr.stats.sampling_rate
        low_f_crit = freqmin / f_nyquist
        high_f_crit = freqmax / f_nyquist
        if high_f_crit - 1.0 > -1e-6:
            raise util.NyquistException(freqmax, f_nyquist, tr.id)

        # Pre-process and apply filter
        tr.detrend("linear")
        tr.taper(0.05, "cosine")
        tr.filter(type="bandpass", freqmin=freqmin, freqmax=freqmax,
                  corners=corners, zerophase=False)

        # Generate filter coefficients for the bandpass filter we
        # applied; this is how the filter is designed within ObsPy
        filter_sos = iirfilter(N=corners, Wn=[low_f_crit, high_f_crit],
                               btype="bandpass", ftype="butter",
                               output="sos")

        return filter_sos

    def _highpass_filter(self, tr):
        """
        Apply a highpass filter to the supplied `obspy.Trace` object; filter
        operation is applied in-place.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace to be filtered.

        Returns
        -------
        filter_sos : ndarray
            Second-order sections representation of the applied filter.

        """

        # Check if we have been diverted here from trying to apply a bandpass
        # filter. Else use specified params for highpass.
        if self.bandpass_filter:
            filt_freq = self.bandpass_lowcut
        else:
            filt_freq = self.highpass_freq
        corners = self.filter_corners

        f_nyquist = 0.5 * tr.stats.sampling_rate
        f_crit = filt_freq / f_nyquist

        # Pre-process and apply filter
        tr.detrend("linear")
        tr.taper(0.05, "cosine")
        tr.filter(type="highpass", freq=filt_freq, corners=corners,
                  zerophase=False)

        # Generate filter coefficients for the highpass filter we
        # applied; this is how the filter is designed within ObsPy
        filter_sos = iirfilter(N=corners, Wn=f_crit, btype="highpass",
                               ftype="butter", output="sos")

        return filter_sos

    def _get_amplitude_windows(self, station, i, event, p_ttimes, s_ttimes,
                               fraction_tt):
        """
        Calculate the start and end time of the windows to measure the max P-
        and S- wave amplitudes in. This is done on the basis of the pick times,
        the event marginal window, the traveltime and uncertainty and the
        specified S-wave signal window.

        P_window_start : P_pick - marginal_window - traveltime_uncertainty
        P_window_end : equivalent to start, or S_pick time; whichever is
                       earlier
        S_window_start : same as P
        S_window_end : S_pick + signal_window + marginal_window +
                       traveltime_uncertainty

        traveltime_uncertainty = traveltime * fraction_tt
            (where fraction_tt is as specified for the lookup table).

        Parameters
        ----------
        station : str
            Station name.
        i : int
            Iterator variable.
        event : `QMigrate.io.Event` object
            Light class encapsulating signal, onset, pick and location
            information for a given event.
        p_ttimes : array-like
            Array of interpolated P traveltimes to the requested grid position.
        s_ttimes : array-like
            Array of interpolated S traveltimes to the requested grid position.
        fraction_tt : float
            An estimate of the uncertainty in the velocity model as a function
            of the traveltime.

        Returns
        -------
        windows : array-like
            [[P_window_start, P_window_end], [S_window_start, S_window_end]]
        picked : bool
            Whether at least one of the phases was picked by the autopicker.

        Raises
        ------
        PickOrderException
            If the P pick for an event/station is later than the S pick.

        """

        p_pick, s_pick, picked = self._get_picks(i, event)

        # Check p_pick is before s_pick
        try:
            assert p_pick < s_pick
        except AssertionError:
            raise util.PickOrderException(event.uid, station, p_pick, s_pick)

        # For P:
        p_start = p_pick - event.marginal_window - p_ttimes[i] * fraction_tt
        p_end = p_pick + event.marginal_window + p_ttimes[i] * fraction_tt
        # For S:
        s_start = s_pick - event.marginal_window - s_ttimes[i] * fraction_tt
        s_end = s_pick + event.marginal_window + s_ttimes[i] * fraction_tt + \
            self.signal_window

        if s_start < p_end:
            if p_end < s_pick:
                windows = [[p_start, p_end],
                           [p_end, s_end]]
            else:
                windows = [[p_start, s_pick - event.marginal_window / 2],
                           [s_pick - event.marginal_window / 2, s_end]]
        else:
            windows = [[p_start, s_start],
                       [s_start, s_end]]

        return windows, picked

    def _get_picks(self, i, event):
        """
        Get picks from this station for this event. Phase picks are preferred,
        but modelled times according to the event location and travel-time
        look-up table are used if no automatic pick was made.

        Parameters
        ----------
        i : int
            Iterator variable.
        event : `QMigrate.io.Event` object
            Light class encapsulating signal, onset, pick and location
            information for a given event.

        Returns
        -------
        p_pick : `obspy.UTCDateTime` object
            P pick time. Autopick time if available, otherwise ModelledTime
            calculated by the autopicker.
        s_pick : `obspy.UTCDateTime` object
            S pick time. Autopick time if available, otherwise ModelledTime
            calculated by the autpicker.
        picked : bool
            Whether at least one of the phases was picked by the autopicker.

        """

        picks = event.picks["df"]
        p_pick = picks.iloc[i*2]["PickTime"]
        s_pick = picks.iloc[i*2+1]["PickTime"]

        picked = True
        # If neither P nor S is picked, picked=False
        if p_pick == -1 and s_pick == -1:
            picked = False
        if p_pick == -1:
            p_pick = picks.iloc[i*2]["ModelledTime"]
        if s_pick == -1:
            s_pick = picks.iloc[i*2+1]["ModelledTime"]

        # Convert to UTCDateTime objects
        p_pick = UTCDateTime(p_pick)
        s_pick = UTCDateTime(s_pick)

        return p_pick, s_pick, picked

    def _measure_signal_amps(self, amps, tr, windows, filter_sos=None):
        """
        Loop through the windows and measure the maximum half peak-to-peak
        amplitude, the approximate frequency (derived from the p2p time) and
        the time at which it occurs.

        Performs a linear detrend across the window before measuring
        amplitudes.

        NOTE: signal amplitudes are converted to *millimetres*. This may have
        to be adjusted depending on the chosen formulation of the local
        magnitude attenuation function.

        Parameters
        ----------
        amps : list
            Amplitude information for this trace.
            Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time",
                       "S_amp", "S_freq", "S_time", "Noise_amp", "is_picked"]
        tr : `obspy.Trace` object
            Trace from which to measure amplitudes.
        windows : array-like
            [[P_window_start, P_window_end], [S_window_start, S_window_end]]
        filter_sos : ndarray, optional
            Second-order sections representation of the filter applied to the
            trace (if applicable).

        Returns
        -------
        amps : list
            Amplitude information for this trace.
            Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time",
                       "S_amp", "S_freq", "S_time", "Noise_amp", "is_picked"]
            With newly populated values:
            P_amp : float
                Half maximum peak-to-trough amplitude in the P signal window.
                In *millimetres*.
            P_freq : float
                Approximate frequency of the maximum amplitude P-wave signal.
                Calculated from the peak-to-trough time of the max
                peak-to-trough amplitude.
            P_time : `obspy.UTCDateTime` object
                Approximate time of amplitude observation (halfway between peak
                and trough times).
            S_amp : float
                As for P but in the S signal window.
            S_freq : float
                As for P.
            S_time : `obspy.UTCDateTime` object
                As for P.
        filter_gain : float or None
            The gain of the filter applied to the trace at the approximate
            frequency of the S-wave p2p signal, if a filter was chosen.
            Else None.

        """

        # Loop over windows, cut data and measure amplitude
        for k, (start_time, end_time) in enumerate(windows):
            window = tr.slice(start_time, end_time)
            window.detrend('linear')
            data = window.data
            phase = ['P', 'S'][k]

            # if trace (window) is empty (no data points) or a flat line, do
            # not make a measurement
            if not bool(window) or data.max() == data.min():
                logging.warning(f"{phase} signal window doesn't contain any "
                                f"data for trace {window.id}")
                continue

            # Measure peak-to-peak amplitude
            try:
                half_amp, approx_freq, p2t_time = \
                    self._peak_to_trough_amplitude(window)
            except util.PeakToTroughError as e:
                logging.warning(f"Amplitude measurement failed in {phase} "
                                f"signal window for trace {window.id}: {e.msg}")
                continue

            # Correct for filter gain at approximate frequency of
            # measured amplitude
            if self.bandpass_filter or self.highpass_filter:
                _, filter_gain = sosfreqz(filter_sos, worN=[approx_freq],
                                          fs=tr.stats.sampling_rate)
                half_amp /= np.abs(filter_gain[0])
            else:
                filter_gain = None

            # Multiply amplitude by 1000 to convert to *millimetres*
            half_amp *= 1000.

            # Put in relevant columns for P / S amplitude, approx_freq,
            # p2t_time
            amps[3+k*3:6+k*3] = half_amp, approx_freq, p2t_time

        return amps, filter_gain

    def _peak_to_trough_amplitude(self, trace):
        """
        Measure the maximum peak-to-trough amplitude for a given trace;
        additionally output the approximate frequency of this signal (from
        the peak-to-trough time) and the time at which it occurs.

        NOTE: Returns *half* the maximum peak-to-trough amplitude, as this is
        what the measurement of local magnitude is defined from.
        NOTE: Units are nanometres; this may have to be adjusted based on the
        chosen formulation of the local magnitude attenuation function.

        Parameters
        ----------
        trace : `obspy.Trace` object
            Waveform for which to measure max peak-to-trough amplitude.

        Returns
        -------
        half_amp : float
            Half the value of maximum peak-to-trough amplitude, in nanometres.
            Returns -1 if no measurement could be made.
        approx_freq : float
            Approximate frequency of the arrival, based on the half-period
            between the maximum peak/trough.
            Returns -1 if no measurement could be made.
        p2t_time : `obspy.UTCDateTime` object
            Approximate time of amplitude observation (halfway between peak and
            trough times.)

        Raises
        ------
        PeakToTroughError
            If the measurement fails, due to no peaks or troughs being found or
            consecutive peaks or troughs being found.

        """

        prominence = self.prominence_multiplier * np.max(np.abs(trace.data))
        peaks, _ = find_peaks(trace.data, prominence=prominence)
        troughs, _ = find_peaks(-trace.data, prominence=prominence)

        # Loop through possible orders of peaks and troughs to find the maximum
        # peak-to-peak amplitude, and the time difference separating the peaks
        full_amp = None
        if len(peaks) == 0 or len(troughs) == 0:
            raise util.PeakToTroughError("No peaks or troughs found!")
        elif len(peaks) == 1 and len(troughs) == 1:
            full_amp = np.abs(trace.data[peaks] - trace.data[troughs])[0]
            pos = 0
        elif len(peaks) == len(troughs):
            if peaks[0] < troughs[0]:
                a, b, c, d = peaks, troughs, peaks[1:], troughs[:-1]
            else:
                a, b, c, d = peaks, troughs, peaks[:-1], troughs[1:]
        elif not np.abs(len(peaks) - len(troughs)) == 1:
            # More than two peaks/troughs next to one another
            raise util.PeakToTroughError("Consecutive peaks/troughs!")
        elif len(peaks) > len(troughs):
            try:
                assert peaks[0] < troughs[0]
            except AssertionError:
                raise util.PeakToTroughError("Consecutive peaks/troughs!")
            a, b, c, d = peaks[:-1], troughs, peaks[1:], troughs
        elif len(peaks) < len(troughs):
            try:
                assert peaks[0] > troughs[0]
            except AssertionError:
                raise util.PeakToTroughError("Consecutive peaks/troughs!")
            a, b, c, d = peaks, troughs[1:], peaks, troughs[:-1]

        if not full_amp:
            fp1 = np.abs(trace.data[a] - trace.data[b])
            fp2 = np.abs(trace.data[c] - trace.data[d])
            if np.max(fp1) >= np.max(fp2):
                pos = np.argmax(fp1)
                full_amp = np.max(fp1)
                peaks, troughs = a, b
            else:
                pos = np.argmax(fp2)
                full_amp = np.max(fp2)
                peaks, troughs = c, d

        peak_time = trace.times()[peaks[pos]]
        trough_time = trace.times()[troughs[pos]]
        p2t_time = trace.stats.starttime + peak_time + \
            (trough_time - peak_time) / 2

        # Peak-to-trough is half a period
        approx_freq = 1. / (np.abs(peak_time - trough_time) * 2.)

        # Local magnitude is defined based on maximum zero-to-peak amplitude
        half_amp = full_amp / 2

        return half_amp, approx_freq, p2t_time

    def _measure_noise_amp(self, tr, windows, filter_gain=None, method="RMS"):
        """
        Make a measurement of the signal amplitude in a 'noise window' before
        the P signal window. Several methods for making this measurement are
        available.

        Performs a linear detrend across the window before measuring
        amplitudes.

        NOTE: Returns the noise amplitude in millimetres: this may have to be
        adjusted depending on the chosen formulation of the local magnitude
        attenuation function.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace from which to measure the noise amplitude.
        windows : array-like
            [[P_window_start, P_window_end], [S_window_start, S_window_end]]
        filter_gain : float, optional
            The gain of the filter applied to the trace at the approximate
            frequency of the S-wave p2p signal, if a filter was chosen.
            (Default None)
        method : {"RMS", "STD"}, optional
            The method by which to measure the amplitude of the signal in the
            noise window: root-mean-square, or standard deviation.
            (Default "RMS")

        Returns
        -------
        noise_amp : float
            An estimate of the signal amplitude in the noise window. In
            millimetres.

        Raises
        ------
        NotImplementedError
            For measurement methods other than {"RMS", "STD"}.

        """

        p_start = windows[0][0]

        # Make a noise measurement in a window of length noise_window,
        # ending at the start of the p_window
        noise_start = p_start - self.noise_window
        noise_end = p_start

        noise = tr.slice(noise_start, noise_end)
        noise.detrend("linear")

        # Use standard deviation in noise window as an estimate of the
        # background noise amplitude *in millimetres*
        if method == "RMS":
            noise_amp = np.sqrt(np.mean(np.square(noise.data))) * 1000.
        elif method == "STD":
            noise_amp = np.std(noise.data) * 1000.
        else:
            raise NotImplementedError("Only 'RMS' and 'STD' are available "
                                      "currently. Please contact the "
                                      "QuakeMigrate developers.")

        if self.bandpass_filter or self.highpass_filter:
            # NOTE: uses the gain at the approx_freq of the last amplitude
            # measured; S-wave signal window.
            noise_amp /= np.abs(filter_gain[0])

        return noise_amp
