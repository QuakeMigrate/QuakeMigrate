# -*- coding: utf-8 -*-
"""
The default seismic phase picking class - fits a 1-D Gaussian to the calculated
onset functions.

"""

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
import pandas as pd
from scipy.optimize import curve_fit

from QMigrate.signal.pick import PhasePicker
import QMigrate.util as util


class GaussianPicker(PhasePicker):
    """
    QuakeMigrate default pick function class.

    Attributes
    ----------
    pick_threshold : float (between 0 and 1)
        For use with picking_mode = 'Gaussian'. Picks will only be made if
        the onset function exceeds this percentile of the noise level
        (average amplitude of onset function outside pick windows).
        Recommended starting value: 1.0

    fraction_tt : float
        Defines width of time window around expected phase arrival time in
        which to search for a phase pick as a function of the travel-time
        from the event location to that station -- should be an estimate of
        the uncertainty in the velocity model.

    phase_picks : dict
        With keys:
            "Pick" : pandas DataFrame
                Phase pick times with columns: ["Name", "Phase",
                                                "ModelledTime",
                                                "PickTime", "PickError",
                                                "SNR"]
                Each row contains the phase pick from one station/phase.

            "GAU_P" : array-like
                Numpy array stack of Gaussian pick info (each as a dict)
                for P phase

            "GAU_S" : array-like
                Numpy array stack of Gaussian pick info (each as a dict)
                for S phase

    Methods
    -------
    pick_phases()
        Picks phase arrival times for located earthquakes by fitting a 1-D
        Gaussian function to the P and S onset functions

    """

    DEFAULT_GAUSSIAN_FIT = {"popt": 0,
                            "xdata": 0,
                            "xdata_dt": 0,
                            "PickValue": -1}

    def __init__(self, onset=None):
        """Class initialisation method."""

        self.onset = onset

        # Pick related parameters
        self.pick_threshold = 1.0
        self.fraction_tt = 0.1

        self.data = None
        self.event = None
        self.times = None
        self.p_ttime = None
        self.s_ttime = None
        self.phase_picks = None

    def __str__(self):
        """
        Return short summary string of the Pick object

        It will provide information on all of the various parameters that the
        user can/has set.

        """

        out = "\tPick parameters - using the 1-D Gaussian fit to onset\n"
        out += "\t\tPick threshold = {}\n"
        out += "\t\tSearch window  = {}s\n\n"
        out = out.format(self.pick_threshold, self.fraction_tt)

        return out

    def pick_phases(self, waveform_data, event):
        """
        Picks phase arrival times for located earthquakes.

        Parameters
        ----------
        event : pandas DataFrame
            Contains data about located event.
            Columns: ["DT", "COA", "X", "Y", "Z"] - X and Y as lon/lat; Z in m

        """

        # If an Onset object has been provided to the picker, recalculate the
        # onset functions for the data
        if self.onset is not None:
            _ = self.onset.calculate_onsets(waveform_data, log=False)

        self.data = waveform_data
        self.event = event

        # start_time and end_time are start of pre-pad and end of post-pad,
        # respectively.
        tmp = np.arange(self.data.start_time,
                        self.data.end_time + self.data.sample_size,
                        self.data.sample_size)
        self.times = pd.to_datetime([x.datetime for x in tmp])

        event_ijk = self.lut.index2coord(event[["X", "Y", "Z"]].values,
                                         inverse=True)[0]

        self.p_ttime = self.lut.traveltime_to("P", event_ijk)
        self.s_ttime = self.lut.traveltime_to("S", event_ijk)

        # Determining the stations that can be picked on and the phases
        picks = pd.DataFrame(index=np.arange(0, 2 * len(self.data.p_onset)),
                             columns=["Name", "Phase", "ModelledTime",
                                      "PickTime", "PickError", "SNR"])

        p_gauss = np.array([])
        s_gauss = np.array([])
        for i, station in self.lut.station_data.iterrows():
            p_arrival = event["DT"] + self.p_ttime[i]
            s_arrival = event["DT"] + self.s_ttime[i]
            idx = 2*i

            for phase in ["P", "S"]:
                if phase == "P":
                    onset = self.data.p_onset[i]
                    arrival = p_arrival
                else:
                    onset = self.data.s_onset[i]
                    arrival = s_arrival
                    idx += 1

                gau, max_onset, err, mn = self._gaussian_picker(
                    onset, phase, self.data.start_time, p_arrival,
                    s_arrival, self.p_ttime[i], self.s_ttime[i])

                if phase == "P":
                    p_gauss = np.hstack([p_gauss, gau])
                else:
                    s_gauss = np.hstack([s_gauss, gau])

                picks.iloc[idx] = [station["Name"], phase, arrival, mn, err,
                                   max_onset]

        phase_picks = {}
        phase_picks["Pick"] = picks
        phase_picks["GAU_P"] = p_gauss
        phase_picks["GAU_S"] = s_gauss

        self.phase_picks = phase_picks

    def write_picks(self, event_uid):
        """
        Write phase picks to a new .picks file

        Parameters
        ----------
        event_uid : str
            event ID for file naming

        """

        self.output.write_picks(self.phase_picks["Pick"], event_uid)

    def _gaussian_picker(self, onset, phase, start_time, p_arr, s_arr, ptt, stt):
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
            Onset (characteristic) function

        phase : str
            Phase name ("P" or "S")

        start_time : UTCDateTime object
            Start time of data (w_beg)

        p_arr : UTCDateTime object
            Time when P-phase is expected to arrive based on best location

        s_arr : UTCDateTime object
            Time when S-phase is expected to arrive based on best location

        ptt : UTCDateTime object
            Traveltime of P-phase

        stt : UTCDateTime object
            Traveltime of S-phase

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

        # Determine indices of P and S pick times
        pt_idx = int((p_arr - start_time) * self.sampling_rate)
        st_idx = int((s_arr - start_time) * self.sampling_rate)

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
        pp_ttime = ptt * self.fraction_tt
        ps_ttime = stt * self.fraction_tt

        # Add length of marginal window to this. Convert to index.
        P_idxmin_new = int(pt_idx - int((self.marginal_window + pp_ttime)
                                        * self.sampling_rate))
        P_idxmax_new = int(pt_idx + int((self.marginal_window + pp_ttime)
                                        * self.sampling_rate))
        S_idxmin_new = int(st_idx - int((self.marginal_window + ps_ttime)
                                        * self.sampling_rate))
        S_idxmax_new = int(st_idx + int((self.marginal_window + ps_ttime)
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
            sta_winlen = self.onset.p_onset_win[0]
            win_min = P_idxmin
            win_max = P_idxmax
        if phase == "S":
            sta_winlen = self.onset.s_onset_win[0]
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
            data_half_range = int(sta_winlen * self.sampling_rate / 2)

            # Select data to fit the gaussian to
            x_data = np.arange(gau_idxmin, gau_idxmax, dtype=float)
            x_data = x_data / self.sampling_rate
            y_data = onset[gau_idxmin:gau_idxmax]

            # Convert indices to times
            x_data_dt = np.array([])
            for i in range(len(x_data)):
                x_data_dt = np.hstack([x_data_dt, start_time + x_data[i]])

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
                mean = start_time + float(popt[1])
                sigma = np.absolute(popt[2])

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

        return gaussian_fit, max_onset, sigma, mean

    def plot(self, file_str=None, event_uid=None, run_path=None):
        """
        Plot figure showing the filtered traces for each data component and the
        characteristic functions calculated from them (P and S) for each
        station. The search window to make a phase pick is displayed, along
        with the dynamic pick threshold (defined as a percentile of the
        background noise level), the phase pick time and its uncertainty (if
        made) and the Gaussian fit to the characteristic function.

        Parameters
        ----------
        file_str : str, optional
            String {run_name}_{evt_id} (figure displayed by default)

        event_uid : str, optional
            Earthquake UID string; for subdirectory naming within directory
            {run_path}/traces/

        """

        # Make output dir for this event outside of loop
        if file_str:
            subdir = "locate/traces/{}".format(event_uid)
            util.make_directories(run_path, subdir=subdir)
            out_dir = run_path / subdir

        # Looping through all stations
        for i in range(self.data.signal.shape[1]):
            station = self.lut.station_data["Name"][i]
            gau_p = self.phase_picks["GAU_P"][i]
            gau_s = self.phase_picks["GAU_S"][i]
            signal = self.data.filtered_signal
            fig = plt.figure(figsize=(30, 15))

            # Defining the plot
            fig.patch.set_facecolor("white")
            x_trace = plt.subplot(322)
            y_trace = plt.subplot(324)
            z_trace = plt.subplot(321)
            p_onset = plt.subplot(323)
            s_onset = plt.subplot(326)

            # Plotting the traces
            self._plot_signal_trace(x_trace, self.times, signal[0, i, :], -1,
                                    "r")
            self._plot_signal_trace(y_trace, self.times, signal[1, i, :], -1,
                                    "b")
            self._plot_signal_trace(z_trace, self.times, signal[2, i, :], -1,
                                    "g")
            p_onset.plot(self.times, self.data.p_onset[i, :], "r",
                         linewidth=0.5)
            s_onset.plot(self.times, self.data.s_onset[i, :], "b",
                         linewidth=0.5)

            # Defining Pick and Error
            picks = self.phase_picks["Pick"]
            phase_picks = picks[picks["Name"] == station].replace(-1, np.nan)
            phase_picks = phase_picks.reset_index(drop=True)

            for _, pick in phase_picks.iterrows():
                if np.isnan(pick["PickError"]):
                    continue

                pick_time = pick["PickTime"]
                pick_err = pick["PickError"]

                if pick["Phase"] == "P":
                    self._pick_vlines(z_trace, pick_time, pick_err)

                    yy = util.gaussian_1d(gau_p["xdata"],
                                          gau_p["popt"][0],
                                          gau_p["popt"][1],
                                          gau_p["popt"][2])
                    gau_dts = [x.datetime for x in gau_p["xdata_dt"]]
                    p_onset.plot(gau_dts, yy)
                    self._pick_vlines(p_onset, pick_time, pick_err)
                else:
                    self._pick_vlines(y_trace, pick_time, pick_err)
                    self._pick_vlines(x_trace, pick_time, pick_err)

                    yy = util.gaussian_1d(gau_s["xdata"],
                                          gau_s["popt"][0],
                                          gau_s["popt"][1],
                                          gau_s["popt"][2])
                    gau_dts = [x.datetime for x in gau_s["xdata_dt"]]
                    s_onset.plot(gau_dts, yy)
                    self._pick_vlines(s_onset, pick_time, pick_err)

            dt_max = self.event["DT"]
            dt_max = UTCDateTime(dt_max)
            self._ttime_vlines(z_trace, dt_max, self.p_ttime[i])
            self._ttime_vlines(p_onset, dt_max, self.p_ttime[i])
            self._ttime_vlines(y_trace, dt_max, self.s_ttime[i])
            self._ttime_vlines(x_trace, dt_max, self.s_ttime[i])
            self._ttime_vlines(s_onset, dt_max, self.s_ttime[i])

            p_onset.axhline(gau_p["PickThreshold"])
            s_onset.axhline(gau_s["PickThreshold"])

            # Refining the window as around the pick time
            min_t = (dt_max + 0.5 * self.p_ttime[i]).datetime
            max_t = (dt_max + 1.5 * self.s_ttime[i]).datetime

            x_trace.set_xlim([min_t, max_t])
            y_trace.set_xlim([min_t, max_t])
            z_trace.set_xlim([min_t, max_t])
            p_onset.set_xlim([min_t, max_t])
            s_onset.set_xlim([min_t, max_t])

            suptitle = "Trace for Station {} - PPick = {}, SPick = {}"
            suptitle = suptitle.format(station,
                                       gau_p["PickValue"], gau_s["PickValue"])

            fig.suptitle(suptitle)

            if file_str is None:
                plt.show()
            else:
                out_str = out_dir / file_str
                fname = "{}_{}.pdf"
                fname = fname.format(out_str, station)
                plt.savefig(fname)
                plt.close("all")

    def _plot_signal_trace(self, ax, x, y, st_idx, color):
        """
        Plot signal trace.

        Performs a simple check to see if there is any signal data available to
        plot.

        Parameters
        ----------
        ax : matplotlib Axes object
            Axes on which to plot the signal trace.

        x : array-like
            Timestamps for the signal trace.

        y : array-like
            The amplitudes of the signal trace.

        st_idx : int
            Amount to vertically shift the signal trace. Either range ordered
            or ordered alphabetically by station name.

        color : str
            Line colour for the trace - see matplotlib documentation for more
            details.

        """

        if y.any():
            ax.plot(x, y / np.max(abs(y)) + (st_idx + 1), color=color,
                    linewidth=0.5, zorder=1)

    def _pick_vlines(self, trace, pick_time, pick_err):
        """
        Plot vlines showing phase pick time and uncertainty.

        """

        trace.axvline((pick_time - pick_err/2).datetime, linestyle="--")
        trace.axvline((pick_time + pick_err/2).datetime, linestyle="--")
        trace.axvline((pick_time).datetime)

    def _ttime_vlines(self, trace, dt_max, ttime):
        """
        Plot vlines showing expected arrival times based on max
        coalescence location.

        """

        trace.axvline((dt_max + ttime).datetime, color="red")
        trace.axvline((dt_max + 0.9 * ttime - self.marginal_window).datetime,
                      color="red", linestyle="--")
        trace.axvline((dt_max + 1.1 * ttime + self.marginal_window).datetime,
                      color="red", linestyle="--")
