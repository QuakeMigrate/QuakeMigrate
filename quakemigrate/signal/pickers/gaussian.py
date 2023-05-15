# -*- coding: utf-8 -*-
"""
The default seismic phase picking class - fits a 1-D Gaussian to the calculated onset
functions.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from quakemigrate.plot.phase_picks import pick_summary
import quakemigrate.util as util
from .base import PhasePicker


class GaussianPicker(PhasePicker):
    """
    This class details the default method of making phase picks shipped with
    QuakeMigrate, namely fitting a 1-D Gaussian function to the onset function for each
    station and phase.

    Attributes
    ----------
    phase_picks : dict
            "GAU_P" : array-like
                Numpy array stack of Gaussian pick info (each as a dict)
                for P phase
            "GAU_S" : array-like
                Numpy array stack of Gaussian pick info (each as a dict)
                for S phase
    threshold_method : {"MAD", "percentile"}
        Which method to use to calculate the pick threshold; a percentile of the data
        outside the pick windows (e.g. 0.99 = 99th percentile) or a multiple of the
        Median Absolute Deviation of the signal outside the pick windows. Default uses
        the MAD method.
    percentile_pick_threshold : float, optional
        Picks will only be made if the onset function exceeds this percentile of the
        noise level (amplitude of onset function outside pick windows). (Default: 1.0)
    mad_pick_threshold : float, optional
        Picks will only be made if the onset function exceeds its median value plus this
        multiple of the MAD (calculated from the onset data outside the pick windows).
        (Default: 8)
    plot_picks : bool
        Toggle plotting of phase picks.

    Methods
    -------
    pick_phases(event, lut, run)
        Picks phase arrival times for located events by fitting a 1-D Gaussian function
        to the P and/or S onset functions

    """

    DEFAULT_GAUSSIAN_FIT = {"popt": 0, "xdata": 0, "xdata_dt": 0, "PickValue": -1}

    def __init__(self, onset=None, **kwargs):
        """Instantiate the GaussianPicker object."""
        super().__init__(**kwargs)

        self.onset = onset

        # --- Get pick method and threshold ---
        self.threshold_method = kwargs.get("threshold_method", "MAD")
        if self.threshold_method == "percentile":
            self.percentile_pick_threshold = kwargs.get(
                "percentile_pick_threshold", 1.0
            )
        elif self.threshold_method == "MAD":
            self.mad_pick_threshold = kwargs.get("mad_pick_threshold", 8.0)
        else:
            raise util.InvalidPickThresholdMethodException
        # Handle deprecated `pick_threshold`
        if kwargs.get("pick_threshold"):
            self.pick_threshold = kwargs["pick_threshold"]

        self.plot_picks = kwargs.get("plot_picks", False)

        if "fraction_tt" in kwargs.keys():
            print(
                "FutureWarning: Fraction of traveltime argument moved to lookup tables."
                "\nIt remains possible to override the fraction of traveltime here, if "
                "required, to further\ntune the phase picker."
            )
        self._fraction_tt = kwargs.get("fraction_tt")

    def __str__(self):
        """Returns a short summary string of the GaussianPicker."""

        str_ = "\tPhase picking by fitting a 1-D Gaussian to onsets\n"
        if self.threshold_method == "percentile":
            str_ += f"\t\tPercentile threshold  = {self.percentile_pick_threshold}\n"
        elif self.threshold_method == "MAD":
            str_ += f"\t\tMAD multiplier  = {self.mad_pick_threshold}\n"
        if self._fraction_tt is not None:
            str_ += f"\t\tSearch window   = {self._fraction_tt*100}% of traveltime\n"

        return str_

    @util.timeit("info")
    def pick_phases(self, event, lut, run):
        """
        Picks phase arrival times for located events.

        Parameters
        ----------
        event : :class:`~quakemigrate.io.event.Event` object
            Light class encapsulating waveforms, coalescence information and location
            information for a given event.
        lut : :class:`~quakemigrate.lut.lut.LUT` object
            Contains the traveltime lookup tables for seismic phases, computed for some
            pre-defined velocity model.
        run : :class:`~quakemigrate.io.core.Run` object
            Light class encapsulating i/o path information for a given run.

        Returns
        -------
        event : :class:`~quakemigrate.io.event.Event` object
            Event object provided to pick_phases(), but now with phase picks!
        picks : `pandas.DataFrame`
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.

        """

        # Onsets are recalculated without logging
        _, onset_data = self.onset.calculate_onsets(
            event.data, log=False, timespan=4 * event.marginal_window
        )

        if self._fraction_tt is None:
            fraction_tt = lut.fraction_tt
        else:
            fraction_tt = self._fraction_tt

        e_ijk = lut.index2coord(event.hypocentre, inverse=True)[0]

        # Pre-define pick DataFrame and fit params and pick windows dicts
        p_idx = np.arange(sum([len(v) for _, v in onset_data.onsets.items()]))
        picks = pd.DataFrame(
            index=p_idx,
            columns=[
                "Station",
                "Phase",
                "ModelledTime",
                "PickTime",
                "PickError",
                "SNR",
            ],
        )
        gaussfits = {}
        pick_windows = {}
        idx = 0

        for station, onsets in onset_data.onsets.items():
            for phase, onset in onsets.items():
                traveltime = lut.traveltime_to(phase, e_ijk, station)[0]
                pick_windows.setdefault(station, {}).update(
                    {
                        phase: self._determine_window(
                            event, onset_data, traveltime, fraction_tt
                        )
                    }
                )
                n_samples = len(onset)

            self._distinguish_windows(
                pick_windows[station], list(onsets.keys()), n_samples
            )

            for phase, onset in onsets.items():
                # Find threshold from 'noise' part of onset
                pick_threshold = self._find_pick_threshold(
                    onset, pick_windows[station], self.threshold_method
                )

                logging.debug(f"\t\tPicking {phase} at {station}...")
                fit, *pick = self._fit_gaussian(
                    onset,
                    onset_data.sampling_rate,
                    self.onset.gaussian_halfwidth(phase),
                    onset_data.starttime,
                    pick_threshold,
                    pick_windows[station][phase],
                )

                gaussfits.setdefault(station, {}).update({phase: fit})

                traveltime = lut.traveltime_to(phase, e_ijk, station)[0]
                model_time = event.otime + traveltime
                picks.iloc[idx] = [station, phase, model_time, *pick]
                idx += 1

        event.add_picks(picks, gaussfits=gaussfits, pick_windows=pick_windows)

        self.write(run, event.uid, picks)

        if self.plot_picks:
            logging.info("\t\tPlotting picks...")
            for station, onsets in onset_data.onsets.items():
                traveltimes = [
                    lut.traveltime_to(phase, e_ijk, station)[0]
                    for phase in onsets.keys()
                ]
                self.plot(event, station, onset_data, picks, traveltimes, run)

        return event, picks

    def _determine_window(self, event, onset_data, tt, fraction_tt):
        """
        Determine phase pick window upper and lower bounds based on the event marginal
        window and a set percentage of the phase travel time.

        Parameters
        ----------
        event : :class:`~quakemigrate.io.event.Event` object
            Light class to encapsulate information about an event, including origin
            time, location and waveform data.
        onset_data : :class:`~quakemigrate.signal.onsets.base.OnsetData` object
            Light class encapsulating data generated during onset calculation.
        tt : float
            Traveltime for the requested phase.
        fraction_tt : float
            Defines width of time window around expected phase arrival time in which to
            search for a phase pick as a function of the traveltime from the event
            location to that station -- should be an estimate of the uncertainty in the
            velocity model.

        Returns
        -------
        lower_idx : int
            Index of lower bound for the phase pick window.
        arrival_idx : int
            Index of the modelled phase arrival time.
        upper_idx : int
            Index of upper bound for the phase pick window.

        """

        arrival_idx = util.time2sample(
            event.otime + tt - onset_data.starttime, onset_data.sampling_rate
        )

        # Add length of marginal window to this and convert to index
        samples = util.time2sample(
            tt * fraction_tt + event.marginal_window, onset_data.sampling_rate
        )

        return [arrival_idx - samples, arrival_idx, arrival_idx + samples]

    def _distinguish_windows(self, windows, phases, samples):
        """
        Ensure pick windows do not overlap - if they do, set the upper bound of window
        one and the lower bound of window two to be the midpoint index of the two
        modelled phase arrival times.

        Parameters
        ----------
        windows : dict
            Dictionary of windows with phases as keys.
        phases : list of str
            Phases being migrated.
        samples : int
            Total number of samples in the onset function.

        """

        # Handle first key
        first_idx = windows[phases[0]][0]
        windows[phases[0]][0] = 0 if first_idx < 0 else first_idx

        # Handle keys pairwise
        for p1, p2 in util.pairwise(phases):
            p1_window, p2_window = windows[p1], windows[p2]
            mid_idx = int((p1_window[1] + p2_window[1]) / 2)
            windows[p1][2] = min(mid_idx, p1_window[2])
            windows[p2][0] = max(mid_idx, p2_window[0])

        # Handle last key
        last_idx = windows[phases[-1]][2]
        windows[phases[-1]][2] = samples if last_idx > samples else last_idx

    def _find_pick_threshold(self, onset, windows, method):
        """
        Determine a pick threshold from the onset data outside the pick windows.

        Parameters
        ----------
        onset : `numpy.ndarray` of `numpy.double`
            Onset (characteristic) function.
        windows : list of int
            Indexes of the lower window bound, the phase arrival, and the upper window
            bound.
        method : {"percentile", "MAD"}
            Method used to calculate the pick threshold from the noise data.

        Return
        ------
        pick_threshold : float
            The threshold calculated from the onset data outside the pick windows,
            according to the specified `method`.

        """

        onset_noise = onset.copy()
        for _, window in windows.items():
            onset_noise[window[0] : window[2]] = -1
        # Remove data during pick windows, and data set to 1 (in onset function taper
        # pad windows)
        onset_noise = onset_noise[onset_noise > 1]

        if method == "percentile":
            pick_threshold = np.percentile(
                onset_noise, self.percentile_pick_threshold * 100
            )
        elif method == "MAD":
            med = np.median(onset_noise)
            mad = util.calculate_mad(onset_noise)
            pick_threshold = med + (mad * self.mad_pick_threshold)

        return pick_threshold

    def _fit_gaussian(
        self, onset, sampling_rate, halfwidth, starttime, pick_threshold, window
    ):
        """
        Fit a Gaussian to the onset function in order to make a time pick with an
        associated uncertainty.

        Uses the amplitude and timing of the onset function peak and some knowledge of
        the onset function parameters (e.g. short-term average window length, for the
        :class:`~quakemigrate.signal.onsets.stalta.STALTAOnset`) to make an initial
        estimate of a gaussian fit to the onset function.

        Parameters
        ----------
        onset : `numpy.ndarray` of `numpy.double`
            Onset function.
        sampling_rate : int
            Sampling rate of the onset function.
        halfwidth : float
            Initial estimate for the Gaussian half-width based on some function of the
            onset function parameters.
        starttime : `obspy.UTCDateTime` object
            Timestamp for first sample of the onset function.
        pick_threshold : float
            Value above which to threshold data based on noise.
        window : list of int, [start, arrival, end]
            Indices for the window start, modelled phase arrival, and window end.

        Returns
        -------
        gaussian_fit : dictionary
            Gaussian fit parameters: {"popt": popt,
                                      "xdata": x_data,
                                      "xdata_dt": x_data_dt,
                                      "PickValue": max_onset,
                                      "PickThreshold": pick_threshold}
        max_onset : float
            Amplitude of Gaussian fit to onset function, i.e. the SNR.
        sigma : float
            Sigma of Gaussian fit to onset function, i.e. the pick uncertainty.
        mean : `obspy.UTCDateTime`
            Mean of Gaussian fit to onset function, i.e. the pick time.

        """

        # Trim the onset function in the pick window
        onset_signal = onset[window[0] : window[2]]
        logging.debug(f"\t\t    win_min: {window[0]}, win_max: {window[2]}")

        # Identify the peak in the windowed onset that exceeds this threshold
        # AND contains the maximum value in the window (i.e. the 'true' peak).
        try:
            peak_idxs = self._find_peak(onset_signal, pick_threshold)
            # add an extra sample either side for the curve fitting. This makes the
            # fitting more stable, and guarantees at least 3 samples --> avoids an
            # under-constrained optimisation (3 fitting params).
            padded_peak_idxs = [peak_idxs[0] - 1, peak_idxs[1] + 1]
            padded_peak_idxs = [window[0] + p for p in padded_peak_idxs]
            logging.debug(
                f"\t\t    padded_peak_idxmin: {padded_peak_idxs[0]},"
                f" padded_peak_idxmax: {padded_peak_idxs[1]}"
            )
            x_data = np.arange(*padded_peak_idxs) / sampling_rate
            y_data = onset[padded_peak_idxs[0] : padded_peak_idxs[1]]
        except util.NoOnsetPeak as e:
            logging.debug(e.msg)
            return self._pick_failure(pick_threshold)

        # Try to fit a 1-D Gaussian
        # Initial parameters (p0) are:
        #   height = max value of onset function
        #   mean   = time of max value
        #   sigma  = `halfwidth` - determined from onset function parameters
        p0 = [
            max(y_data),
            (padded_peak_idxs[0] + np.argmax(y_data)) / sampling_rate,
            halfwidth / sampling_rate,
        ]
        try:
            popt, _ = curve_fit(util.gaussian_1d, x_data, y_data, p0)
        except (ValueError, RuntimeError) as e:
            # curve_fit can fail for a number of reasons - primarily if the input data
            # contains nans or if the least-squares minimisation fails. A warning may
            # also be emitted to stdout if the covariance of the parameters could not
            # be estimated - this is suppressed by default in scan.py.
            logging.debug(f"\t\t    Failed curve_fit:\n{e}\n\t\t    Continuing...")
            return self._pick_failure(pick_threshold)
        except TypeError as e:
            logging.debug(
                "\t\t    Failed curve_fit - too few input data?"
                f"{e}\n\t\t    Continuing..."
            )

        # Unpack results:
        #  popt = [height, mean (seconds), sigma (seconds)]
        max_onset = popt[0]
        mean = starttime + float(popt[1])
        sigma = np.absolute(popt[2])

        # Check pick mean is within the pick window.
        if not window[0] < popt[1] * sampling_rate < window[2]:
            logging.debug("\t\t    Pick mean out of bounds - continuing.")
            return self._pick_failure(pick_threshold)

        gaussian_fit = {
            "popt": popt,
            "xdata": x_data,
            "xdata_dt": np.array([starttime + x for x in x_data]),
            "PickValue": max_onset,
            "PickThreshold": pick_threshold,
        }

        return gaussian_fit, mean, sigma, max_onset

    def _pick_failure(self, pick_threshold):
        """
        Short utility function to produce the default values when a pick cannot be made.

        Parameters
        ----------
        pick_threshold : float
            Pick threshold value for onset data.

        Returns
        -------
        gaussian_fit : dictionary
            The default Gaussian fit dictionary, with relevant pick threshold value.
        max_onset : int
            A default of -1 value to indicate failure.
        sigma : int
            A default of -1 value to indicate failure.
        mean : int
            A default of -1 value to indicate failure.

        """

        gaussian_fit = self.DEFAULT_GAUSSIAN_FIT.copy()
        gaussian_fit["PickThreshold"] = pick_threshold
        mean = sigma = max_onset = -1

        return gaussian_fit, mean, sigma, max_onset

    def _find_peak(self, windowed_onset, pick_threshold):
        """
        Identify peaks, if any, within the windowed onset that exceed the specified
        threshold value. Of those peaks, this function seeks the one that contains the
        maximum value within the window, i.e. the 'true' peak - see the diagram below.

                                             v
                                             *
                                            * *
                                    *      *   *
                         |         * *    *     *     |
                         |---------------#-------#----|
                         |        *    **         *   |

        Parameters
        ----------
        windowed_onset : `numpy.ndarray` of `numpy.double`
            The onset function within the picking window.
        pick_threshold : float
            Value above which to search for peaks in the onset data.

        Returns
        -------
        true_peak_idx : [int, int]
            Start and end index values for the 'true' peak, with +1 added to the last
            index so that all of the values above the threshold are returned when
            slicing by index.

        Raises
        ------
        util.NoOnsetPeak
            If no onset data, or only a single sample, exceeds the pick threshold.

        """

        exceedence = np.where(windowed_onset > pick_threshold)[0]
        if len(exceedence) == 0:
            raise util.NoOnsetPeak(pick_threshold)

        # Identify all peaks - there are possibly multiple distinct periods of data that
        # exceed the threshold. The following command simply seeks non-consecutive index
        # values in the array of points that exceed the threshold and splits the array
        # at these points into 'peaks'.
        peaks = np.split(exceedence, np.where(np.diff(exceedence) != 1)[0] + 1)

        # Identify the peak that contains the true peak (maximum)
        true_maximum = np.argmax(windowed_onset)
        for i, peak in enumerate(peaks):
            if np.any(peak == true_maximum):
                break

        # Check if there is more than a single sample above the threshold
        if len(peaks[i]) < 2:
            raise util.NoOnsetPeak(pick_threshold)

        # Grab the peak and return the start/end index values. NOTE: + 1 is required so
        # that the last sample is included when slicing by index
        true_peak_idxs = [peaks[i][0], peaks[i][-1] + 1]

        return true_peak_idxs

    @util.timeit()
    def plot(self, event, station, onset_data, picks_df, traveltimes, run):
        """
        Plot figure showing the filtered traces for each data component and the onset
        functions calculated from them (P and/or S) for each station. The search window
        to make a phase pick is displayed, along with the dynamic pick threshold, the
        phase pick time and its uncertainty (if made) and the Gaussian fit to the onset
        function.

        Parameters
        ----------
        event : :class:`~quakemigrate.io.event.Event` object
            Light class to encapsulate information about an event, including origin
            time, location and waveform data.
        station : str
            Station name.
        onset_data : :class:`~quakemigrate.signal.onsets.base.OnsetData` object
            Light class encapsulating data generated during onset calculation.
        picks_df : `pandas.DataFrame` object
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.
        traveltimes : list of float
            Modelled traveltimes from the event hypocentre to the station for each phase
            to be plotted.
        run : :class:`~quakemigrate.io.core.Run` object
            Light class encapsulating i/o path information for a given run.

        """

        fpath = run.path / f"locate/{run.subname}/pick_plots/{event.uid}"
        fpath.mkdir(exist_ok=True, parents=True)

        onsets = onset_data.onsets[station]
        waveforms = onset_data.filtered_waveforms.select(station=station)
        # Check if any data available to plot
        if not bool(waveforms):
            return
        picks = picks_df[picks_df["Station"] == station].reset_index(drop=True)
        windows = event.picks["pick_windows"][station]

        # Call subroutine to plot phase pick figure
        fig = pick_summary(
            event, station, waveforms, picks, onsets, traveltimes, windows
        )

        fstem = f"{event.uid}_{station}"
        file = (fpath / fstem).with_suffix(".pdf")
        plt.savefig(file)
        plt.close(fig)

    @property
    def fraction_tt(self):
        """Handler for deprecated attribute 'fraction_tt'"""
        return self._fraction_tt

    @fraction_tt.setter
    def fraction_tt(self, value):
        print(
            "FutureWarning: Fraction of traveltime attribute has moved to lookup table."
            "\nOverriding..."
        )
        self._fraction_tt = value

    @property
    def pick_threshold(self):
        """Handler for deprecated attribute 'pick_threshold'"""

    @pick_threshold.setter
    def pick_threshold(self, value):
        raise AttributeError(
            "The 'pick_threshold' attribute has been deprecated. Select a threshold "
            "method from 'percentile' or 'MAD', and see the docs for the syntax for "
            "the appropriate threshold."
        )
