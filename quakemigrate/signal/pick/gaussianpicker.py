# -*- coding: utf-8 -*-
"""
The default seismic phase picking class - fits a 1-D Gaussian to the calculated
onset functions.

:copyright:
    2020, QuakeMigrate developers.
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
from .pick import PhasePicker


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
        self.marginal_window = kwargs.get("marginal_window", 1.0)
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
                f"\t\tPick threshold  = {self.pick_threshold}\n"
                f"\t\tMarginal window = {self.marginal_window} s\n")
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
        _ = self.onset.calculate_onsets(event.data, log=False)

        if self._fraction_tt is None:
            fraction_tt = lut.fraction_tt
        else:
            fraction_tt = self._fraction_tt

        e_ijk = lut.index2coord(event.hypocentre, inverse=True)[0]
        ptt = lut.traveltime_to("P", e_ijk)
        stt = lut.traveltime_to("S", e_ijk)

        # Pre-define pick DataFrame and fit params and pick windows dicts
        picks = pd.DataFrame(index=np.arange(0, 2 * len(event.data.p_onset)),
                             columns=["Station", "Phase", "ModelledTime",
                                      "PickTime", "PickError", "SNR"])
        gaussfits = {}
        pick_windows = {}

        for i, station in lut.station_data.iterrows():
            gaussfits[station["Name"]] = {}
            pick_windows[station["Name"]] = {}
            for j, phase in enumerate(["P", "S"]):
                if phase == "P":
                    onset = event.data.p_onset[i]
                    model_time = event.otime + ptt[i]
                else:
                    onset = event.data.s_onset[i]
                    model_time = event.otime + stt[i]

                gau, max_onset, pick, pick_error, window = self._fit_gaussian(
                    onset, phase, event.data.starttime, event.otime,
                    ptt[i], stt[i], fraction_tt)

                gaussfits[station["Name"]][phase] = gau
                pick_windows[station["Name"]][phase] = window

                picks.iloc[2*i+j] = [station["Name"], phase, model_time, pick,
                                     pick_error, max_onset]

        event.add_picks(picks, gaussfits=gaussfits, pick_windows=pick_windows,
                        pick_threshold=self.pick_threshold)

        self.write(run, event.uid, picks)

        if self.plot_picks:
            logging.info("\t\tPlotting picks...")
            self.plot(event, lut, picks, list(zip(ptt, stt)), run)

        return event, picks

    def _fit_gaussian(self, onset, phase, starttime, otime, ptt, stt,
                      fraction_tt):
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
        onset : array-like
            Onset (characteristic) function.
        phase : str
            Phase name ("P" or "S").
        starttime : UTCDateTime object
            Start time of data (w_beg).
        p_arr : UTCDateTime object
            Time when P phase is expected to arrive based on best location.
        s_arr : UTCDateTime object
            Time when S phase is expected to arrive based on best location.
        ptt : UTCDateTime object
            Traveltime of P phase.
        stt : UTCDateTime object
            Traveltime of S phase.
        fraction_tt : float
            Defines width of time window around expected phase arrival time in
            which to search for a phase pick as a function of the traveltime
            from the event location to that station -- should be an estimate of
            the uncertainty in the velocity model.

        Returns
        -------
        gaussian_fit : dictionary
            gaussian fit parameters: {"popt": popt,
                                      "xdata": x_data,
                                      "xdata_dt": x_data_dt,
                                      "PickValue": max_onset,
                                      "PickThreshold": threshold}
        max_onset : float
            amplitude of gaussian fit to onset function
        sigma : float
            sigma of gaussian fit to onset function
        mean : UTCDateTime
            mean of gaussian fit to onset function == pick time

        """

        p_arr, s_arr = otime + ptt, otime + stt

        # Determine indices of P and S pick times
        pt_idx = int((p_arr - starttime) * self.sampling_rate)
        st_idx = int((s_arr - starttime) * self.sampling_rate)

        # Determine P and S pick window upper and lower bounds based on
        # (P-S)/2 -- either this or the next window definition will be
        # used depending on which is wider.
        pmin_idx = int(pt_idx - (st_idx - pt_idx) / 2)
        pmax_idx = int(pt_idx + (st_idx - pt_idx) / 2)
        smin_idx = int(st_idx - (st_idx - pt_idx) / 2)
        smax_idx = int(st_idx + (st_idx - pt_idx) / 2)

        # Check if index falls outside length of onset function; if so set
        # window to start/end at start/end of data.
        for idx in [pmin_idx, pmax_idx, smin_idx, smax_idx]:
            if idx < 0:
                idx = 0
            if idx > len(onset):
                idx = len(onset)

        # Defining the bounds to search for the event over
        # Determine P and S pick window upper and lower bounds based on
        # set percentage of total travel time, plus marginal window

        # window based on self.fraction_tt of P/S travel time
        ptt *= fraction_tt
        stt *= fraction_tt

        # Add length of marginal window to this. Convert to index.
        P_idxmin_new = int(pt_idx - int((self.marginal_window + ptt)
                                        * self.sampling_rate))
        P_idxmax_new = int(pt_idx + int((self.marginal_window + ptt)
                                        * self.sampling_rate))
        S_idxmin_new = int(st_idx - int((self.marginal_window + stt)
                                        * self.sampling_rate))
        S_idxmax_new = int(st_idx + int((self.marginal_window + stt)
                                        * self.sampling_rate))

        # Setting so the search region can't be bigger than (P-S)/2:
        # compare the two window definitions; if (P-S)/2 window is
        # smaller then use this (to avoid picking the wrong phase).
        P_idxmin = np.max([pmin_idx, P_idxmin_new])
        P_idxmax = np.min([pmax_idx, P_idxmax_new])
        S_idxmin = np.max([smin_idx, S_idxmin_new])
        S_idxmax = np.min([smax_idx, S_idxmax_new])

        # Setting parameters depending on the phase
        if phase == "P":
            win_min = P_idxmin
            win_max = P_idxmax
        if phase == "S":
            win_min = S_idxmin
            win_max = S_idxmax

        # Find index of maximum value of onset function in the appropriate
        # pick window
        max_onset = np.argmax(onset[win_min:win_max]) + win_min
        # Trim the onset function in the pick window
        onset_trim = onset[win_min:win_max]

        # Only keep the onset function outside the pick windows to
        # calculate the pick threshold
        onset_threshold = onset.copy()
        onset_threshold[P_idxmin:P_idxmax] = -1
        onset_threshold[S_idxmin:S_idxmax] = -1
        onset_threshold = onset_threshold[onset_threshold > -1]

        # Calculate the pick threshold: either user-specified percentile of
        # data outside pick windows, or 88th percentile within the relevant
        # pick window (whichever is bigger).
        threshold = np.percentile(onset_threshold, self.pick_threshold * 100)
        threshold_window = np.percentile(onset_trim, 88)
        threshold = np.max([threshold, threshold_window])

        # Remove data within the pick window that is lower than the threshold
        tmp = (onset_trim - threshold).any() > 0

        # If there is any data that meets this requirement...
        if onset[max_onset] >= threshold and tmp:
            exceedence = np.where((onset_trim - threshold) > 0)[0]
            exceedence_dist = np.zeros(len(exceedence))

            # Really faffy process to identify the period of data which is
            # above the threshold around the highest value of the onset
            # function.
            d = 1
            e = 0
            while e < len(exceedence_dist) - 1:
                if e == len(exceedence_dist):
                    exceedence_dist[e] = d
                else:
                    if exceedence[e + 1] == exceedence[e] + 1:
                        exceedence_dist[e] = d
                    else:
                        exceedence_dist[e] = d
                        d += 1
                e += 1

            # Find the indices for this period of data
            tmp = exceedence_dist[np.argmax(onset_trim[exceedence])]
            tmp = np.where(exceedence_dist == tmp)

            # Add one data point below the threshold at each end of this period
            gau_idxmin = exceedence[tmp][0] + win_min - 1
            gau_idxmax = exceedence[tmp][-1] + win_min + 2

            # Initial guess for gaussian half-width based on onset function
            # STA window length
            data_half_range = self.onset.gaussian_halfwidth(phase)

            # Select data to fit the gaussian to
            x_data = np.arange(gau_idxmin, gau_idxmax, dtype=float)
            x_data = x_data / self.sampling_rate
            y_data = onset[gau_idxmin:gau_idxmax]

            # Convert indices to times
            x_data_dt = np.array([])
            for _, x in enumerate(x_data):
                x_data_dt = np.hstack([x_data_dt, starttime + x])

            # Try to fit a 1-D Gaussian.
            try:
                # Initial parameters are:
                #  height = max value of onset function
                #  mean   = time of max value
                #  sigma  = data half-range (calculated above)
                p0 = [np.max(y_data),
                      float(gau_idxmin + np.argmax(y_data))
                      / self.sampling_rate,
                      data_half_range / self.sampling_rate]

                # Do the fit
                popt, _ = curve_fit(util.gaussian_1d, x_data, y_data, p0)

                # Results:
                #  popt = [height, mean (seconds), sigma (seconds)]
                max_onset = popt[0]
                # Convert mean (pick time) to time
                mean = starttime + float(popt[1])
                sigma = np.absolute(popt[2])

                # Check pick mean is within the pick window.
                if not gau_idxmin < popt[1] * self.sampling_rate < gau_idxmax:
                    gaussian_fit = self.DEFAULT_GAUSSIAN_FIT
                    gaussian_fit["PickThreshold"] = threshold
                    sigma = -1
                    mean = -1
                    max_onset = -1
                else:
                    gaussian_fit = {"popt": popt,
                                    "xdata": x_data,
                                    "xdata_dt": x_data_dt,
                                    "PickValue": max_onset,
                                    "PickThreshold": threshold}

            # If curve_fit fails. Will also spit error message to stdout,
            # though this can be suppressed  - see warnings.filterwarnings()
            except (ValueError, RuntimeError):
                gaussian_fit = self.DEFAULT_GAUSSIAN_FIT
                gaussian_fit["PickThreshold"] = threshold
                sigma = -1
                mean = -1
                max_onset = -1

        # If onset function does not exceed threshold in pick window
        else:
            gaussian_fit = self.DEFAULT_GAUSSIAN_FIT
            gaussian_fit["PickThreshold"] = threshold
            sigma = -1
            mean = -1
            max_onset = -1

        return gaussian_fit, max_onset, mean, sigma, [win_min, win_max]

    @util.timeit()
    def plot(self, event, lut, picks, ttimes, run):
        """
        Plot figure showing the filtered traces for each data component and the
        characteristic functions calculated from them (P and S) for each
        station. The search window to make a phase pick is displayed, along
        with the dynamic pick threshold (defined as a percentile of the
        background noise level), the phase pick time and its uncertainty (if
        made) and the Gaussian fit to the characteristic function.

        Parameters
        ----------
        event_uid : str, optional
            Earthquake UID string; for subdirectory naming within directory
            {run_path}/traces/

        """

        fpath = run.path / "locate" / run.subname / "pick_plots" / f"{event.uid}"
        fpath.mkdir(exist_ok=True, parents=True)

        # Generate plottable timestamps for data
        times = event.data.times(type="matplotlib")
        for i, station in lut.station_data["Name"].iteritems():
            signal = event.data.filtered_signal[:, i, :]
            onsets = [event.data.p_onset[i, :], event.data.s_onset[i, :]]
            stpicks = picks[picks["Station"] == station].reset_index(drop=True)
            window = event.picks["pick_windows"][station]

            # Check if any data available to plot
            if not signal.any():
                continue

            # Call subroutine to plot basic phase pick figure
            fig = pick_summary(event, station, signal, stpicks, onsets,
                               ttimes[i], window)

            # --- Gaussian fits ---
            axes = fig.axes
            for j, (ax, ph) in enumerate(zip(axes[3:5], ["P", "S"])):
                gau = event.picks["gaussfits"][station][ph]
                win = window[ph]

                # Plot threshold
                thresh = gau["PickThreshold"]
                norm = max(onsets[j][win[0]:win[1]+1])
                ax.axhline(thresh / norm, label="Pick threshold")
                axes[5].text(0.05+j*0.5, 0.25, f"Threshold: {thresh:5.3f}",
                             ha="left", va="center", fontsize=8)

                # Check pick has been made
                if not gau["PickValue"] == -1:
                    yy = util.gaussian_1d(gau["xdata"], gau["popt"][0],
                                          gau["popt"][1], gau["popt"][2])
                    dt = [x.datetime for x in gau["xdata_dt"]]
                    ax.plot(dt, yy / norm)

            # --- Picking windows ---
            for j, ax in enumerate(axes[:5]):
                win = window["P"] if j % 3 == 0 else window["S"]
                clr = "#F03B20" if j % 3 == 0 else "#3182BD"
                ax.fill_betweenx([-1.1, 1.1], times[win[0]], times[win[1]],
                                 alpha=0.2, color=clr, label="Picking window")

            for ax in axes[3:5]:
                ax.legend(fontsize=8)

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
