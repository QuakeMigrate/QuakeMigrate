# -*- coding: utf-8 -*-
"""
The default seismic phase picking class - fits a 1-D Gaussian to the calculated
onset functions.

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
    QuakeMigrate, namely fitting a 1-D Gaussian function to the STA/LTA onset
    function trace for each station.

    Attributes
    ----------
    phase_picks : dict
            "GAU_P" : array-like
                Numpy array stack of Gaussian pick info (each as a dict)
                for P phase
            "GAU_S" : array-like
                Numpy array stack of Gaussian pick info (each as a dict)
                for S phase
    pick_threshold : float (between 0 and 1)
        Picks will only be made if the onset function exceeds this percentile
        of the noise level (average amplitude of onset function outside pick
        windows). Recommended starting value: 1.0
    plot_picks : bool
        Toggle plotting of phase picks.

    Methods
    -------
    pick_phases(data, lut, event, event_uid, output)
        Picks phase arrival times for located earthquakes by fitting a 1-D
        Gaussian function to the P and S onset functions

    """

    DEFAULT_GAUSSIAN_FIT = {"popt": 0,
                            "xdata": 0,
                            "xdata_dt": 0,
                            "PickValue": -1}

    def __init__(self, onset=None, **kwargs):
        """Instantiate the GaussianPicker object."""
        super().__init__(**kwargs)

        self.onset = onset
        self.pick_threshold = kwargs.get("pick_threshold", 1.0)
        self.sampling_rate = None
        self.plot_picks = kwargs.get("plot_picks", False)

        if "fraction_tt" in kwargs.keys():
            print("FutureWarning: Fraction of traveltime argument moved to "
                  "lookup tables.\nIt remains possible to override the "
                  "fraction of traveltime here, if required, to further\ntune"
                  "the phase picker.")
        self._fraction_tt = kwargs.get("fraction_tt")

    def __str__(self):
        """Returns a short summary string of the GaussianPicker."""

        str_ = ("\tPhase picking by fitting a 1-D Gaussian to onsets\n"
                f"\t\tPick threshold  = {self.pick_threshold}\n")
        if self._fraction_tt is not None:
            str_ += (f"\t\tSearch window   = {self._fraction_tt*100}% of "
                     "traveltime\n")

        return str_

    @util.timeit("info")
    def pick_phases(self, event, lut, run):
        """
        Picks phase arrival times for located earthquakes.

        Parameters
        ----------
        event : :class:`~quakemigrate.io.event.Event` object
            Contains pre-processed waveform data on which to perform picking,
            the event location, and a unique identifier.
        lut : :class:`~quakemigrate.lut.LUT` object
            Contains the traveltime lookup tables for seismic phases, computed
            for some pre-defined velocity model.
        run : :class:`~quakemigrate.io.Run` object
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
        _ = self.onset.calculate_onsets(event.data, lut.phases, log=False)
        self.sampling_rate = self.onset.sampling_rate

        if self._fraction_tt is None:
            fraction_tt = lut.fraction_tt
        else:
            fraction_tt = self._fraction_tt

        e_ijk = lut.index2coord(event.hypocentre, inverse=True)[0]

        # Pre-define pick DataFrame and fit params and pick windows dicts
        p_idx = np.arange(sum([len(v) for _, v in event.data.onsets.items()]))
        picks = pd.DataFrame(index=p_idx,
                             columns=["Station", "Phase", "ModelledTime",
                                      "PickTime", "PickError", "SNR"])
        gaussfits = {}
        pick_windows = {}
        idx = 0

        for station, onsets in event.data.onsets.items():
            for phase, onset in onsets.items():
                traveltime = lut.traveltime_to(phase, e_ijk, station)[0]
                pick_windows.setdefault(station, {}).update(
                    {phase: self._determine_window(
                        event, traveltime, fraction_tt)})
                n_samples = len(onset)

            self._distinguish_windows(
                pick_windows[station], list(onsets.keys()), n_samples)

            for phase, onset in onsets.items():
                # Find threshold from 'noise' part of onset
                noise_threshold = self._find_noise_threshold(
                    onset, pick_windows[station])

                logging.debug(f"\t\tPicking {phase} at {station}...")
                fit, *pick = self._fit_gaussian(
                    onset, self.onset.gaussian_halfwidth(phase),
                    event.data.starttime, noise_threshold,
                    pick_windows[station][phase])

                gaussfits.setdefault(station, {}).update({phase: fit})

                traveltime = lut.traveltime_to(phase, e_ijk, station)[0]
                model_time = event.otime + traveltime
                picks.iloc[idx] = [station, phase, model_time, *pick]
                idx += 1

        event.add_picks(picks, gaussfits=gaussfits, pick_windows=pick_windows,
                        pick_threshold=self.pick_threshold)

        self.write(run, event.uid, picks)

        if self.plot_picks:
            logging.info("\t\tPlotting picks...")
            for station, onsets in event.data.onsets.items():
                traveltimes = [lut.traveltime_to(phase, e_ijk, station)[0]
                               for phase in onsets.keys()]
                self.plot(event, station, onsets, picks, traveltimes, run)

        return event, picks

    def _determine_window(self, event, tt, fraction_tt):
        """
        Determine phase pick window upper and lower bounds based on a set
        percentage of the phase travel time.

        Parameters
        ----------
        event : :class:`~quakemigrate.io.event.Event` object
            Contains pre-processed waveform data on which to perform picking,
            the event location, and a unique identifier.
        tt : float
            Traveltime for the requested phase.
        fraction_tt : float
            Defines width of time window around expected phase arrival time in
            which to search for a phase pick as a function of the traveltime
            from the event location to that station -- should be an estimate of
            the uncertainty in the velocity model.

        Returns
        -------
        lower_idx : int
            Index of lower bound for the phase pick window.
        arrival_idx : int
            Index of the phase arrival.
        upper_idx : int
            Index of upper bound for the phase pick window.

        """

        arrival_idx = util.time2sample(event.otime + tt - event.data.starttime,
                                       self.sampling_rate)

        # Add length of marginal window to this and convert to index
        samples = util.time2sample(tt * fraction_tt + event.marginal_window,
                                   self.sampling_rate)

        return [arrival_idx - samples, arrival_idx, arrival_idx + samples]

    def _distinguish_windows(self, windows, phases, samples):
        """
        Ensure pick windows do not overlap - if they do, set the upper bound of
        window one and the lower bound of window two to be the midpoint index
        of the two arrivals.

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
            mid_idx = int((p1_window[2] + p2_window[0]) / 2)
            windows[p1][2] = min(mid_idx, p1_window[2])
            windows[p2][0] = max(mid_idx, p2_window[0])

        # Handle last key
        last_idx = windows[phases[-1]][2]
        windows[phases[-1]][2] = samples if last_idx > samples else last_idx

    def _find_noise_threshold(self, onset, windows):
        """
        Determine a pick threshold as a percentile of the onset data outside
        the pick windows.

        Parameters
        ----------
        onset : `numpy.ndarray` of `numpy.double`
            Onset (characteristic) function.
        windows : list of int
            Indexes of the lower window bound, the phase arrival, and the upper
            window bound.

        Return
        ------
        noise_threshold : float
            The threshold based on 'noise'.

        """

        onset_noise = onset.copy()
        for _, window in windows.items():
            onset_noise[window[0]:window[2]] = -1
        onset_noise = onset_noise[onset_noise > -1]

        noise_threshold = np.percentile(onset_noise, self.pick_threshold * 100)

        return noise_threshold

    def _fit_gaussian(self, onset, halfwidth, starttime, noise_threshold,
                      window):
        """
        Fit a Gaussian to the onset function in order to make a time pick with
        an associated uncertainty. Uses the same STA/LTA onset (characteristic)
        function as is migrated through the grid to calculate the earthquake
        location.

        Uses knowledge of approximate pick index, the short-term average
        onset window and the signal sampling rate to make an initial estimate
        of a gaussian fit to the onset function.

        Parameters
        ----------
        onset : `numpy.ndarray` of `numpy.double`
            Onset (characteristic) function.
        halfwidth : float
            Initial estimate for the Gaussian half-width based on the
            short-term average window length.
        starttime : `obspy.UTCDateTime` object
            Timestamp for first sample of data.
        noise_threshold : float
            Value above which to threshold data based on noise.
        window : list of int, [start, arrival, end]
            Indices for the window start, phase arrival, and window end.

        Returns
        -------
        gaussian_fit : dictionary
            Gaussian fit parameters: {"popt": popt,
                                      "xdata": x_data,
                                      "xdata_dt": x_data_dt,
                                      "PickValue": max_onset,
                                      "PickThreshold": threshold}
        max_onset : float
            Amplitude of Gaussian fit to onset function, i.e. the SNR.
        sigma : float
            Sigma of Gaussian fit to onset function, i.e. the pick uncertainty.
        mean : `obspy.UTCDateTime`
            Mean of Gaussian fit to onset function, i.e. the pick time.

        """

        # Trim the onset function in the pick window
        onset_signal = onset[window[0]:window[2]]

        # Calculate the pick threshold: either user-specified percentile of
        # data outside pick windows, or 88th percentile within the relevant
        # pick window (whichever is bigger).
        signal_threshold = np.percentile(onset_signal, 88)
        threshold = np.max([noise_threshold, signal_threshold])

        # Identify the peak in the windowed onset that exceeds this threshold
        # AND contains the maximum value in the window (i.e. the 'true' peak).
        try:
            peak_idxs = self._find_peak(onset_signal, threshold)
            # add an extra sample either side for the curve fitting. This makes
            # the fitting more stable, and guarantees at least 3 samples -->
            # avoids an under-constrained optimisation (3 fitting params).
            padded_peak_idxs = [peak_idxs[0] - 1, peak_idxs[1] + 1]
            padded_peak_idxs = [window[0] + p for p in padded_peak_idxs]
            x_data = np.arange(*padded_peak_idxs) / self.sampling_rate
            y_data = onset[padded_peak_idxs[0]:padded_peak_idxs[1]]
        except util.NoOnsetPeak as e:
            logging.debug(e.msg)
            return self._pick_failure(threshold)

        # Try to fit a 1-D Gaussian
        # Initial parameters (p0) are:
        #   height = max value of onset function
        #   mean   = time of max value
        #   sigma  = data half-range
        p0 = [max(y_data),
              (padded_peak_idxs[0] + np.argmax(y_data)) / self.sampling_rate,
              halfwidth / self.sampling_rate]
        try:
            popt, _ = curve_fit(util.gaussian_1d, x_data, y_data, p0)
        except (ValueError, RuntimeError) as e:
            # curve_fit can fail for a number of reasons - primarily if the
            # input data contains nans or if the least-squares minimisation
            # fails. A warning may also be emitted to stdout if the covariance
            # of the parameters could not be estimated - this is suppressed by
            # default in scan.py.
            logging.debug(f"\t\t    Failed curve_fit:\n{e}\n\t\t    "
                          "Continuing...")
            return self._pick_failure(threshold)
        except TypeError as e:
            logging.debug("\t\t    Failed curve_fit - too few input data?"
                          f"{e}\n\t\t    Continuing...")

        # Unpack results:
        #  popt = [height, mean (seconds), sigma (seconds)]
        max_onset = popt[0]
        mean = starttime + float(popt[1])
        sigma = np.absolute(popt[2])

        # Check pick mean is within the pick window.
        peak_idxs = [window[0] + p for p in peak_idxs]
        if not peak_idxs[0] < popt[1] * self.sampling_rate < peak_idxs[1]:
            logging.debug("\t\t    Pick mean out of bounds - continuing.")
            return self._pick_failure(threshold)

        gaussian_fit = {"popt": popt,
                        "xdata": x_data,
                        "xdata_dt": np.array([starttime + x for x in x_data]),
                        "PickValue": max_onset,
                        "PickThreshold": threshold}

        return gaussian_fit, mean, sigma, max_onset

    def _pick_failure(self, threshold):
        """
        Short utility function to produce the default values when a pick cannot
        be made.

        Parameters
        ----------
        threshold : float
            Threshold value for onset data.

        Returns
        -------
        gaussian_fit : dictionary
            The default Gaussian fit dictionary, with relevant threshold value.
        max_onset : int
            A default of -1 value to indicate failure.
        sigma : int
            A default of -1 value to indicate failure.
        mean : int
            A default of -1 value to indicate failure.

        """

        gaussian_fit = self.DEFAULT_GAUSSIAN_FIT.copy()
        gaussian_fit["PickThreshold"] = threshold
        mean = sigma = max_onset = -1

        return gaussian_fit, mean, sigma, max_onset

    def _find_peak(self, windowed_onset, threshold):
        """
        Identify peaks, if any, within the windowed onset that exceed the
        specified threshold value. Of those peaks, this function seeks the one
        that contains the maximum value within the window, i.e. the 'true'
        peak - see the diagram below.

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
        threshold : float
            Value above which to search for peaks in the onset data.

        Returns
        -------
        true_peak_idx : [int, int]
            Start and end index values for the 'true' peak, with +1 added to
            the last index so that all of the values above the threshold are
            returned when slicing by index.

        Raises
        ------
        util.NoOnsetPeak
            If no onset data, or only a single sample, exceeds the threshold.

        """

        exceedence = np.where(windowed_onset > threshold)[0]
        if len(exceedence) < 2:
            raise util.NoOnsetPeak(threshold)

        # Identify all peaks - there are possibly multiple distinct periods
        # of data that exceed the threshold. The following command simply seeks
        # non-consecutive index values in the array of points that exceed
        # the threshold and splits the array at these points into 'peaks'.
        peaks = np.split(exceedence, np.where(np.diff(exceedence) != 1)[0] + 1)

        # Identify the peak that contains the true peak (maximum)
        true_maximum = np.argmax(windowed_onset)
        for i, peak in enumerate(peaks):
            if np.any(peak == true_maximum):
                break

        # Grab the peak and return the start/end index values. NOTE: + 1 is
        # required so that the last sample is included when slicing by index
        true_peak_idxs = [peaks[i][0], peaks[i][-1] + 1]

        return true_peak_idxs

    @util.timeit()
    def plot(self, event, station, onsets, picks, traveltimes, run):
        """
        Plot figure showing the filtered traces for each data component and the
        characteristic functions calculated from them (P and S) for each
        station. The search window to make a phase pick is displayed, along
        with the dynamic pick threshold (defined as a percentile of the
        background noise level), the phase pick time and its uncertainty (if
        made) and the Gaussian fit to the characteristic function.

        Parameters
        ----------
        event : :class:`~quakemigrate.io.Event` object
            Light class encapsulating signal, onset, and location information
            for a given event.
        station : str
            Station name.
        onsets : `numpy.ndarray`
            Collection of onset functions to be plotted.
        picks : `pandas.DataFrame` object
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.
        traveltimes : list of float
            Modelled traveltimes for the station/phase pairs to be plotted.
        run : :class:`~quakemigrate.io.Run` object
            Light class encapsulating i/o path information for a given run.

        """

        fpath = run.path / f"locate/{run.subname}/pick_plots/{event.uid}"
        fpath.mkdir(exist_ok=True, parents=True)

        signal = event.data.filtered_waveforms.select(station=station)
        # Check if any data available to plot
        if not bool(signal):
            return
        stpicks = picks[picks["Station"] == station].reset_index(drop=True)
        window = event.picks["pick_windows"][station]

        # Call subroutine to plot basic phase pick figure
        fig = pick_summary(event, station, signal, stpicks, onsets,
                           traveltimes, window)

        # --- Gaussian fits ---
        axes = fig.axes
        phases = [phase for phase, _ in onsets.items()]
        onsets = [onset for _, onset in onsets.items()]
        for j, (ax, ph) in enumerate(zip(axes[3:5], phases)):
            gau = event.picks["gaussfits"][station][ph]
            win = window[ph]

            # Plot threshold
            thresh = gau["PickThreshold"]
            norm = max(onsets[j][win[0]:win[2]+1])
            ax.axhline(thresh / norm, label="Pick threshold")
            axes[5].text(0.05+j*0.5, 0.25, f"Threshold: {thresh:5.3f}",
                         ha="left", va="center", fontsize=18)

            # Check pick has been made
            if not gau["PickValue"] == -1:
                yy = util.gaussian_1d(gau["xdata"], gau["popt"][0],
                                      gau["popt"][1], gau["popt"][2])
                dt = [x.datetime for x in gau["xdata_dt"]]
                ax.plot(dt, yy / norm)

        # --- Picking windows ---
        # Generate plottable timestamps for data
        times = event.data.times(type="matplotlib")
        for j, ax in enumerate(axes[:5]):
            win = window[phases[0]] if j % 3 == 0 else window[phases[-1]]
            clr = "#F03B20" if j % 3 == 0 else "#3182BD"
            ax.fill_betweenx([-1.1, 1.1], times[win[0]], times[win[2]],
                             alpha=0.2, color=clr, label="Picking window")

        for ax in axes[3:5]:
            ax.legend(fontsize=14)

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
        print("FutureWarning: Fraction of traveltime attribute has moved to "
              "lookup table.\n Overriding...")
        self._fraction_tt = value
