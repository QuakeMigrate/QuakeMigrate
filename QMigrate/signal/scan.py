# -*- coding: utf-8 -*-
"""
Module to perform QuakeMigrate detect and locate

"""

import warnings

import numpy as np
from obspy import UTCDateTime
from obspy.signal.invsim import cosine_taper
from obspy.signal.trigger import classic_sta_lta
import pandas as pd
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, fftconvolve

import QMigrate.core.model as cmod
import QMigrate.core.QMigratelib as ilib
import QMigrate.io.quakeio as qio
import QMigrate.plot.quakeplot as qplot
import QMigrate.util as util

# Catch warnings as errors
warnings.filterwarnings("always")


def sta_lta_centred(a, nsta, nlta):
    """
    Calculates the ratio of the average signal of a short-term window to a
    long-term window.

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
    sta / lta : float
        Ratio of short term average and long term average

    """

    nsta = int(nsta)
    nlta = int(nlta)

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
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] - dtiny

    return sta / lta


def onset(sig, stw, ltw, centred=False):
    """
    Define an onset function

    Parameters
    ----------
    sig : array-like
        Data signal used to generate an onset function

    stw : float
        Short term window size

    ltw : float
        Long term window size

    Returns
    -------
    onset_raw : array-like
        Copy of onset function generated from data

    onset : array-like
        Onset function generated from data

    """

    n_channels, n_samples = sig.shape
    onset = np.copy(sig)
    onset_raw = np.copy(sig)
    for i in range(n_channels):
        if np.sum(sig[i, :]) == 0.0:
            onset[i, :] = 0.0
            onset_raw[i, :] = onset[i, :]
        else:
            if centred is True:
                onset[i, :] = sta_lta_centred(sig[i, :], stw, ltw)
            else:
                onset[i, :] = classic_sta_lta(sig[i, :], stw, ltw)
            onset_raw[i, :] = onset[i, :]
            np.clip(1 + onset[i, :], 0.8, np.inf, onset[i, :])
            np.log(onset[i, :], onset[i, :])

    return onset_raw, onset


def filter(sig, sampling_rate, lc, hc, order=3):
    """
    Filter seismic data

    Parameters
    ----------
    sig : array-like
        Data signal to be filtered

    sampling_rate : int
        Number of samples per second, in Hz

    lc : float
        Lowpass frequency of filter

    hc : float
        Highpass frequency of filter

    order : int
        Number of corners

    """

    b1, a1 = butter(order, [2.0 * lc / sampling_rate,
                            2.0 * hc / sampling_rate], btype="band")
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)
    # sig = detrend(sig)
    for ch in range(0, nchan):
        fsig[ch, :] = fsig[ch, :] - fsig[ch, 0]
        tap = cosine_taper(len(fsig[ch, :]), 0.1)
        fsig[ch, :] = fsig[ch, :] * tap
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, ::-1])[::-1]
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, :])

    return fsig


class DefaultSeisScan(object):
    """Default parameter class for SeisScan"""

    def __init__(self):
        """Initialise object"""

        # Filter parameters
        self.p_bp_filter = [2.0, 16.0, 2]
        self.s_bp_filter = [2.0, 12.0, 2]

        # Onset window parameters
        self.p_onset_win = [0.2, 1.0]
        self.s_onset_win = [0.2, 1.0]

        # Traveltime lookup table decimation factor
        self.decimate = [1, 1, 1]

        # Time step for continuous compute in detect
        self.time_step = 120

        # Data sampling rate
        self.sampling_rate = 100.0

        # Centred onset function override
        self.onset_centred = None

        # Pick related parameters
        self.pick_threshold = 1.0
        self.picking_mode = "Gaussian"
        self.percent_tt = 0.1

        # Marginal window - essentially an estimate of the traveltime error in
        # the LUT
        self.marginal_window = 2

        # Default pre-pad for compute
        self.pre_pad = None

        # Number of cores to perform detect/locate on
        self.n_cores = 1

        # Plotting toggles
        self.plot_coal_grid = False
        self.plot_coal_video = False
        self.plot_coal_summary = True
        self.plot_coal_trace = False

        # xy files for plotting
        self.xy_files = None


class SeisScan(DefaultSeisScan):
    """
    QuakeMigrate scanning class

    Forms the core of the QuakeMigrate method, providing wrapping functions for
    the C-compiled methods.

    Attributes
    ----------
    pre_pad : float

    post_pad : float
        Maximum travel-time from a point in the grid to a station

    Methods
    -------
    detect(start_time, end_time, log=False)
        Core detection method

    locate(start_time, end_time, cut_mseed=False, log=False)
        Core locate method

    """

    raw_data = {}
    filt_data = {}
    onset_data = {}

    DEFAULT_GAUSSIAN_FIT = {"popt": 0,
                            "xdata": 0,
                            "xdata_dt": 0,
                            "PickValue": -1}

    EVENT_FILE_COLS = ["DT", "COA", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ"]

    def __init__(self, data, lookup_table, output_path=None, output_name=None):
        """
        Class initialisation method

        Parameters
        ----------
        data :

        lookup_table :

        output_path :

        output_name :

        """

        DefaultSeisScan.__init__(self)

        self.data = data
        lut = cmod.LUT()
        lut.load(lookup_table)
        self.lut = lut

        if output_path is not None:
            self.output = qio.QuakeIO(output_path, output_name)
        else:
            self.output = None

        ttmax = np.max(lut.fetch_map("TIME_S"))
        self.post_pad = round(ttmax + ttmax*0.05)

        msg = "=" * 120 + "\n"
        msg += "=" * 120 + "\n"
        msg += "   QuakeMigrate - Coalescence Scanning - Path: {} - Name: {}\n"
        msg += "=" * 120 + "\n"
        msg += "=" * 120 + "\n"
        msg = msg.format(self.output.path, self.output.name)
        print(msg)

    def __str__(self):
        """
        Return short summary string of the SeisScan object

        It will provide information on all of the various parameters that the
        user can/has set.

        """

        out = "QuakeMigrate parameters"
        out += "\n\tTime step\t\t:\t{}".format(self.time_step)
        out += "\n\n\tData sampling rate\t:\t{}".format(self.sampling_rate)
        out += "\n\tOutput sampling rate\t:\t{}".format(
            self.output_sampling_rate)
        out += "\n\n\tDecimation\t\t:\t[{}, {}, {}]".format(
            self.decimate[0], self.decimate[1], self.decimate[2])
        out += "\n\n\tBandpass filter P\t:\t[{}, {}, {}]".format(
            self.p_bp_filter[0], self.p_bp_filter[1], self.p_bp_filter[2])
        out += "\n\tBandpass filter S\t:\t[{}, {}, {}]".format(
            self.s_bp_filter[0], self.s_bp_filter[1], self.s_bp_filter[2])
        out += "\n\n\tOnset P [STA, LTA]\t:\t[{}, {}]".format(
            self.p_onset_win[0], self.p_onset_win[1])
        out += "\n\tOnset S [STA, LTA]\t:\t[{}, {}]".format(
            self.s_onset_win[0], self.s_onset_win[1])
        out += "\n\n\tPre-pad\t\t\t:\t{}".format(self.pre_pad)
        out += "\n\tPost-pad\t\t:\t{}".format(self.post_pad)
        out += "\n\n\tMarginal window\t\t:\t{}".format(self.marginal_window)
        out += "\n\tPick threshold\t\t:\t{}".format(self.pick_threshold)
        out += "\n\tPicking mode\t\t:\t{}".format(self.picking_mode)
        out += "\n\tPercent ttime\t\t:\t{}".format(self.percent_tt)
        out += "\n\n\tCentred onset\t\t:\t{}".format(self.onset_centred)
        out += "\n\n\tNumber of CPUs\t\t:\t{}".format(self.n_cores)

        return out

    def detect(self, start_time, end_time, log=False):
        """
        Scans through continuous data to find earthquakes by calculating
        coalescence on a decimated grid

        Parameters
        ----------
        start_time : str
            Time stamp of first sample

        end_time : str
            Time stamp of final sample

        log : bool, optional
            Output processing to a log file

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        self.log = log

        # Conduct the continuous compute on the decimated grid
        self.lut = self.lut.decimate(self.decimate)

        # Detect uses the non-centred onset by default
        if self.onset_centred is None:
            self.onset_centred = False

        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.p_onset_win[1],
                               self.s_onset_win[1]) \
                           + 3 * max(self.p_onset_win[0],
                                     self.s_onset_win[0])

        # Detect the possible events from the decimated grid
        self._continuous_compute(start_time, end_time)

    def locate(self, start_time, end_time, cut_mseed=False, log=False):
        """
        Evaluates the location of events triggered from the detect stage on a
        non-decimated grid.

        Parameters
        ----------
        start_time : str
            Time stamp of first sample

        end_time : str
            Time stamp of final sample

        cut_mseed : bool, optional
            Saves cut mSEED files if True

        log : bool, optional
            Output processing to a log file

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        self.log = log

        msg = "=" * 120 + "\n"
        msg += "   LOCATE - Determining earthquake location and uncertainty\n"
        msg += "=" * 120 + "\n"
        msg += "\n"
        msg += "   Parameters specified:\n"
        msg += "         Start time                = {}\n"
        msg += "         End   time                = {}\n"
        msg += "         Number of CPUs            = {}\n\n"
        msg += "=" * 120 + "\n"
        msg = msg.format(str(start_time), str(end_time), self.n_cores)
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        events = self.output.read_triggered_events(start_time, end_time)

        n_evts = len(events)

        # Conduct the continuous compute on the decimated grid
        self.lut = self.lut.decimate(self.decimate)

        # Locate uses the centred onset by default
        if self.onset_centred is None:
            self.onset_centred = True

        if self.pre_pad is None:
            self.pre_pad = max(self.p_onset_win[1],
                               self.s_onset_win[1]) \
                           + 3 * max(self.p_onset_win[0],
                                     self.s_onset_win[0])

        # Adjust pre- and post-pad to take into account cosine taper
        t_length = self.pre_pad + 4*self.marginal_window + self.post_pad
        self.pre_pad += round(t_length * 0.06)
        self.post_pad += round(t_length * 0.06)

        for i, event in events.iterrows():
            evt_id = event["EventID"]
            msg = "=" * 120 + "\n"
            msg += "    EVENT - {} of {} - {}\n"
            msg += "=" * 120 + "\n\n"
            msg += "    Determining event location..."
            msg = msg.format(i + 1, n_evts, evt_id)
            if self.log:
                self.output.write_log(msg)
            else:
                print(msg)

            timer = util.Stopwatch()
            print("    Computing 4D coalescence grid")

            w_beg = event["CoaTime"] - 2*self.marginal_window - self.pre_pad
            w_end = event["CoaTime"] + 2*self.marginal_window + self.post_pad

            try:
                self.data.read_mseed(w_beg, w_end, self.sampling_rate)
            except util.ArchiveEmptyException:
                msg = "\tNo files in archive for this time period"
                print(msg)
                continue
            except util.DataGapException:
                msg = "\tAll available data for this time period contains gaps"
                msg += "\n\tor data not available at start/end of time period\n"
                print(msg)
                continue

            daten, dsnr, dsnr_norm, dloc, map_ = self._compute(
                                                    w_beg, w_end,
                                                    self.data.signal,
                                                    self.data.availability)
            dcoord = self.lut.xyz2coord(np.array(dloc).astype(int))

            event_coa_val = pd.DataFrame(np.array((daten, dsnr,
                                                   dcoord[:, 0],
                                                   dcoord[:, 1],
                                                   dcoord[:, 2])).transpose(),
                                         columns=["DT", "COA", "X", "Y", "Z"])
            event_coa_val["DT"] = event_coa_val["DT"].apply(UTCDateTime)
            event_coa_val_dtmax = event_coa_val["DT"].iloc[event_coa_val["COA"].astype("float").idxmax()]
            w_beg_mw = event_coa_val_dtmax - self.marginal_window
            w_end_mw = event_coa_val_dtmax + self.marginal_window

            if (event_coa_val_dtmax >= event["CoaTime"] - self.marginal_window) \
               and (event_coa_val_dtmax <= event["CoaTime"] + self.marginal_window):
                w_beg_mw = event_coa_val_dtmax - self.marginal_window
                w_end_mw = event_coa_val_dtmax + self.marginal_window
            else:
                msg = "\tEvent {} is outside marginal window.\n"
                msg += "\tDefine more realistic error - the marginal window"
                msg += " should be an estimate of the traveltime error in the\n"
                msg += "\tlookup table and velocity model.\n"
                msg = msg.format(evt_id)
                if self.log:
                    self.output.write_log(msg)
                else:
                    print(msg)
                continue

            event = event_coa_val
            event = event[(event["DT"] >= w_beg_mw) & (event["DT"] <= w_end_mw)]
            map_ = map_[:, :, :, event.index[0]:event.index[-1]]
            event = event.reset_index(drop=True)
            event_max = event.iloc[event["COA"].astype("float").idxmax()]

            # Determining the hypocentral location from the maximum over
            # the marginal window.
            picks, GAUP, GAUS = self._arrival_picker(event_max, evt_id)

            station_pick = {}
            station_pick["Pick"] = picks
            station_pick["GAU_P"] = GAUP
            station_pick["GAU_S"] = GAUS
            print(timer())

            # Determining earthquake location error
            timer = util.Stopwatch()
            print("    Determining earthquake location and uncertainty...")
            loc_spline, loc, loc_err, loc_cov, loc_err_cov = self._location_error(map_)
            print(timer())

            evt = pd.DataFrame([[event_max.values[0], event_max.values[1],
                                 loc_spline[0], loc_spline[1], loc_spline[2],
                                 loc[0], loc[1], loc[2],
                                 loc_err[0], loc_err[1], loc_err[2],
                                 loc_cov[0], loc_cov[1], loc_cov[2],
                                 loc_err_cov[0], loc_err_cov[1],
                                 loc_err_cov[2]]],
                               columns=self.EVENT_FILE_COLS)

            evt_id = str(event_max.values[0])
            for char_ in ["-", ":", ".", " ", "Z", "T"]:
                evt_id = evt_id.replace(char_, "")

            self.output.write_event(evt, evt_id)

            if cut_mseed:
                print("    Creating cut Mini-SEED...")
                timer = util.Stopwatch()
                self.output.cut_mseed(self.data, evt_id)
                print(timer())

            if self.plot_coal_trace:
                timer = util.Stopwatch()
                print("    Creating station traces...")
                seis_plot = qplot.QuakePlot(self.lut, map_, self.coa_map,
                                            self.data, event, station_pick,
                                            self.marginal_window)
                out = str(self.output.run / "traces" / "{}_{}".format(
                    self.output.name,
                    evt_id))
                seis_plot.coalescence_trace(output_file=out)
                del seis_plot
                print(timer())

            if self.plot_coal_grid:
                timer = util.Stopwatch()
                print("    Creating 4D coalescence grids...")
                self.output.write_coal4D(map_, evt_id, w_beg, w_end)
                print(timer())

            if self.plot_coal_video:
                timer = util.Stopwatch()
                print("    Creating seismic videos...")
                seis_plot = qplot.QuakePlot(self.lut, map_, self.coa_map,
                                            self.data, event, station_pick,
                                            self.marginal_window)
                out = str(self.output.run / "videos" / "{}_{}".format(
                    self.output.name,
                    evt_id))
                seis_plot.coalescence_video(output_file=out)
                del seis_plot
                print(timer())

            if self.plot_coal_summary:
                timer = util.Stopwatch()
                print("    Creating overview figure...")
                seis_plot = qplot.QuakePlot(self.lut, map_, self.coa_map,
                                            self.data, event, station_pick,
                                            self.marginal_window)
                out = str(self.output.run / "summaries" / "{}_{}".format(
                    self.output.name,
                    evt_id))
                seis_plot.coalescence_summary(output_file=out,
                                              earthquake=evt)
                del seis_plot
                print(timer())

            print("=" * 120 + "\n")

            del map_, event, station_pick
            self.coa_map = None

    def _continuous_compute(self, start_time, end_time):
        """
        Compute coalescence between two time stamps divided into small time
        steps.

        Parameters
        ----------
        start_time : UTCDateTime object
            Time stamp of first sample

        end_time : UTCDateTime object
            Time stamp of final sample

        """

        coalescence_mSEED = None

        msg = "=" * 120 + "\n"
        msg += "   DETECT - Continuous Seismic Processing\n"
        msg += "=" * 120 + "\n"
        msg += "\n"
        msg += "   Parameters specified:\n"
        msg += "         Start time                = {}\n"
        msg += "         End   time                = {}\n"
        msg += "         Time step (s)             = {}\n"
        msg += "         Number of CPUs            = {}\n"
        msg += "\n"
        msg += "         Sampling rate             = {}\n"
        msg += "         Grid decimation [X, Y, Z] = [{}, {}, {}]\n"
        msg += "         Bandpass filter P         = [{}, {}, {}]\n"
        msg += "         Bandpass filter S         = [{}, {}, {}]\n"
        msg += "         Onset P [STA, LTA]        = [{}, {}]\n"
        msg += "         Onset S [STA, LTA]        = [{}, {}]\n"
        msg += "\n"
        msg += "=" * 120
        msg = msg.format(str(start_time), str(end_time), self.time_step,
                         self.n_cores, self.sampling_rate,
                         self.decimate[0], self.decimate[1], self.decimate[2],
                         self.p_bp_filter[0], self.p_bp_filter[1],
                         self.p_bp_filter[2], self.s_bp_filter[0],
                         self.s_bp_filter[1], self.s_bp_filter[2],
                         self.p_onset_win[0], self.p_onset_win[1],
                         self.s_onset_win[0], self.s_onset_win[1])
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        t_length = self.pre_pad + self.post_pad + self.time_step
        self.pre_pad += round(t_length * 0.06)
        self.post_pad += round(t_length * 0.06)

        try:
            nsteps = int(np.ceil((end_time - start_time) / self.time_step))
        except AttributeError:
            msg = "Time step has not been specified"
            print(msg)

        for i in range(nsteps):
            w_beg = start_time + self.time_step * i - self.pre_pad
            w_end = start_time + self.time_step * (i + 1) + self.post_pad

            msg = ("~" * 24) + " Processing : {} - {} " + ("~" * 24)
            msg = msg.format(str(w_beg), str(w_end))
            if self.log:
                self.output.write_log(msg)
            else:
                print(msg)

            try:
                self.data.read_mseed(w_beg, w_end, self.sampling_rate)
                daten, dsnr, dsnr_norm, dloc, map_ = self._compute(
                                                        w_beg, w_end,
                                                        self.data.signal,
                                                        self.data.availability)

                dcoord = self.lut.xyz2coord(dloc)
                del dloc, map_
            except util.ArchiveEmptyException:
                msg = "!" * 24 + " " * 16
                msg += " No files in archive for this time step "
                msg += " " * 16 + "!" * 24
                print(msg)
                daten, dsnr, dsnr_norm, dcoord = self._empty(w_beg, w_end)
            except util.DataGapException:
                msg = "!" * 24 + " " * 9
                msg += "All available data for this time period contains gaps"
                msg += " " * 10 + "!" * 24
                msg += "\n" + "!" * 24 + " " * 11
                msg += "or data not available at start/end of time period"
                msg += " " * 12 + "!" * 24
                print(msg)
                daten, dsnr, dsnr_norm, dcoord = self._empty(w_beg, w_end)

            coalescence_mSEED = self.output.write_decscan(coalescence_mSEED,
                                                          daten[:-1],
                                                          dsnr[:-1],
                                                          dsnr_norm[:-1],
                                                          dcoord[:-1, :],
                                                          self.sampling_rate)

            del daten, dsnr, dsnr_norm, dcoord

        print("=" * 120)

    def _compute(self, w_beg, w_end, signal, station_availability):
        """
        Compute coalescence between two time stamps

        Parameters
        ----------
        w_beg : UTCDateTime object
            Time stamp of first sample in window

        w_end : UTCDateTime object
            Time stamp of final sample in window

        signal : array-like
            Data stream

        station_availability : array-like
            List of available stations

        Returns
        -------
        daten : array-like
            Array of UTCDateTime time stamps for the time step

        dsnr :
            Coalescence value through time

        dsnr_norm :
            Normalised coalescence value through time

        dloc :
            Location of maximum coalescence through time

        map_ :
            4-D coalescence map through time

        """

        avail_idx = np.where(station_availability == 1)[0]
        sige = signal[0]
        sign = signal[1]
        sigz = signal[2]

        p_onset_raw, p_onset = self._compute_p_onset(sigz,
                                                     self.sampling_rate)
        s_onset_raw, s_onset = self._compute_s_onset(sige, sign,
                                                     self.sampling_rate)
        self.data.p_onset = p_onset
        self.data.s_onset = s_onset
        self.data.p_onset_raw = p_onset_raw
        self.data.s_onset_raw = s_onset_raw

        ps_onset = np.concatenate((self.data.p_onset, self.data.s_onset))
        ps_onset[np.isnan(ps_onset)] = 0

        p_ttime = self.lut.fetch_index("TIME_P", self.sampling_rate)
        s_ttime = self.lut.fetch_index("TIME_S", self.sampling_rate)
        ttime = np.c_[p_ttime, s_ttime]
        del p_ttime, s_ttime

        nchan, tsamp = ps_onset.shape

        pre_smp = int(round(self.pre_pad * int(self.sampling_rate)))
        pos_smp = int(round(self.post_pad * int(self.sampling_rate)))
        nsamp = tsamp - pre_smp - pos_smp

        daten = 0.0 - pre_smp / self.sampling_rate

        ncell = tuple(self.lut.cell_count)

        map_ = np.zeros(ncell + (nsamp,), dtype=np.float64)

        dind = np.zeros(nsamp, np.int64)
        dsnr = np.zeros(nsamp, np.double)

        ilib.scan(ps_onset, ttime, pre_smp, pos_smp, nsamp, map_, self.n_cores)
        ilib.detect(map_, dsnr, dind, 0, nsamp, self.n_cores)

        # Get dsnr_norm
        sum_coa = np.sum(map_, axis=(0, 1, 2))
        dsnr_norm = dsnr / sum_coa
        dsnr_norm = dsnr_norm * map_.shape[0] * map_.shape[1] * map_.shape[2]

        tmp = np.arange(w_beg + self.pre_pad,
                        w_end - self.post_pad + (1 / self.sampling_rate),
                        1 / self.sampling_rate)
        daten = [x.datetime for x in tmp]
        dsnr = np.exp((dsnr / (len(avail_idx) * 2)) - 1.0)
        dloc = self.lut.xyz2index(dind, inverse=True)

        return daten, dsnr, dsnr_norm, dloc, map_

    def _compute_p_onset(self, sig_z, sampling_rate):
        """
        Generates an onset function for the Z-component

        Parameters
        ----------
        sig_z : array-like
            Z-component time-series data

        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        p_onset_raw : array-like
            Onset function generated from raw vertical component data

        p_onset : array-like
            Onset function generated from filtered vertical component data

        """

        stw, ltw = self.p_onset_win
        stw = int(stw * sampling_rate) + 1
        ltw = int(ltw * sampling_rate) + 1
        sig_z = self._preprocess_p(sig_z, sampling_rate)
        self.filt_data["sigz"] = sig_z
        p_onset_raw, p_onset = onset(sig_z, stw, ltw,
                                     centred=self.onset_centred)
        self.onset_data["sigz"] = p_onset

        return p_onset_raw, p_onset

    def _preprocess_p(self, sig_z, sampling_rate):
        """
        Pre-processing method for Z-component

        Applies a butterworth bandpass filter.

        Parameters
        ----------
        sig_z : array-like
            Z-component time-series data

        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        A filtered version of the vertical component time-series data

        """

        lc, hc, ord_ = self.p_bp_filter
        sig_z = filter(sig_z, sampling_rate, lc, hc, ord_)
        self.data.filtered_signal[2, :, :] = sig_z

        return sig_z

    def _compute_s_onset(self, sig_e, sig_n, sampling_rate):
        """
        Generates onset functions for the N- and E-components

        Parameters
        ----------
        sig_e : array-like
            E-component time-series

        sig_n : array-like
            N-component time-series

        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        s_onset_raw : array-like
            Onset function generated from raw horizontal component data

        s_onset : array-like
            Onset function generated from filtered horizontal component data

        """

        stw, ltw = self.s_onset_win
        stw = int(stw * sampling_rate) + 1
        ltw = int(ltw * sampling_rate) + 1
        sig_e, sig_n = self._preprocess_s(sig_e, sig_n, sampling_rate)
        self.filt_data["sige"] = sig_e
        self.filt_data["sign"] = sig_n
        s_e_onset_raw, s_e_onset = onset(sig_e, stw, ltw,
                                         centred=self.onset_centred)
        s_n_onset_raw, s_n_onset = onset(sig_n, stw, ltw,
                                         centred=self.onset_centred)
        self.onset_data["sige"] = s_e_onset
        self.onset_data["sign"] = s_n_onset
        s_onset = np.sqrt((s_e_onset ** 2 + s_n_onset ** 2) / 2.)
        s_onset_raw = np.sqrt((s_e_onset_raw ** 2 + s_n_onset_raw ** 2) / 2.)
        self.onset_data["sigs"] = s_onset

        return s_onset_raw, s_onset

    def _preprocess_s(self, sig_e, sig_n, sampling_rate):
        """
        Pre-processing method for N- and E-components

        Applies a butterworth bandpass filter.

        Parameters
        ----------
        sig_e : array-like
            E-component time-series
        sig_n : array-like
            N-component time-series
        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        A filtered version of the N- and E-components time-series

        """
        lc, hc, ord_ = self.s_bp_filter
        sig_e = filter(sig_e, sampling_rate, lc, hc, ord_)
        sig_n = filter(sig_n, sampling_rate, lc, hc, ord_)
        self.data.filtered_signal[0, :, :] = sig_n
        self.data.filtered_signal[1, :, :] = sig_e

        return sig_e, sig_n

    def _gaussian_picker(self, onset, phase, start_time, p_arr, s_arr, ptt, stt):
        """
        Fit a Gaussian to the onset function in order to make a time pick.

        Uses knowledge of approximate pick index, the short-term average
        onset window and the signal sampling rate.

        Parameters
        ----------
        onset :
            Onset function

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
            gaussian fit parameters

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
        pmin_idx = int(pt_idx - (st_idx - pt_idx) / 2)  # unnecessary?
        pmax_idx = int(pt_idx + (st_idx - pt_idx) / 2)
        smin_idx = int(st_idx - (st_idx - pt_idx) / 2)
        smax_idx = int(st_idx + (st_idx - pt_idx) / 2)  # unnecessary?

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

        # window based on self.percent_tt of P/S travel time
        pp_ttime = ptt * self.percent_tt
        ps_ttime = stt * self.percent_tt

        # Add length of marginal window to this. Convert to index.
        P_idxmin_new = int(pt_idx - int((self.marginal_window + pp_ttime)
                                        * self.sampling_rate))
        P_idxmax_new = int(pt_idx + int((self.marginal_window + pp_ttime)
                                        * self.sampling_rate))
        S_idxmin_new = int(st_idx - int((self.marginal_window + ps_ttime)
                                        * self.sampling_rate))
        S_idxmax_new = int(st_idx + int((self.marginal_window + ps_ttime)
                                        * self.sampling_rate))

        # Setting so the search region can"t be bigger than (P-S)/2.
        # Compare these two window definitions. If (P-S)/2 window is
        # smaller then use this (to avoid picking the wrong phase).
        P_idxmin = np.max([pmin_idx, P_idxmin_new])
        P_idxmax = np.min([pmax_idx, P_idxmax_new])
        S_idxmin = np.max([smin_idx, S_idxmin_new])
        S_idxmax = np.min([smax_idx, S_idxmax_new])

        # Setting parameters depending on the phase
        if phase == "P":
            sta_winlen = self.p_onset_win[0]
            win_min = P_idxmin
            win_max = P_idxmax
        if phase == "S":
            sta_winlen = self.s_onset_win[0]
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

            # Try to fit a gaussian.
            try:
                # Initial parameters are:
                #  height = max value of onset function
                #  mean   = time of max value
                #  sigma  = data half-range (calculated above)
                p0 = [np.max(y_data),
                      float(gau_idxmin + np.argmax(y_data)) / self.sampling_rate,
                      data_half_range / self.sampling_rate]

                # Do the fit
                popt, pcov = curve_fit(util.gaussian_1d, x_data, y_data, p0)

                # Results:
                #  popt = [height, mean (seconds), sigma (seconds)]
                #  pcov not used
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
            except:
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

    def _arrival_picker(self, max_coa, event_name):
        """
        Determines arrival times for triggered earthquakes.

        Parameters
        ----------
        max_coa : pandas DataFrame object
            DataFrame containing the maximum coalescence values for a
            given event

        event_name : str
            Event ID - used for saving the picks file

        Returns
        -------
        picks : pandas DataFrame object
            DataFrame containing refined pick times

        p_gauss : array-like
            Numpy array stack of Gaussian picks for P phase

        s_gauss : array-like
            Numpy array stack of Gaussian picks for S phase

        """

        p_onset = self.data.p_onset
        s_onset = self.data.s_onset
        start_time = self.data.start_time

        max_coa_crd = np.array([max_coa[["X", "Y", "Z"]].values])
        max_coa_xyz = np.array(self.lut.xyz2coord(max_coa_crd,
                                                  inverse=True)).astype(int)[0]

        p_ttime = self.lut.value_at("TIME_P", max_coa_xyz)[0]
        s_ttime = self.lut.value_at("TIME_S", max_coa_xyz)[0]

        # Determining the stations that can be picked on and the phases
        picks = pd.DataFrame(index=np.arange(0, 2 * len(p_onset)),
                             columns=["Name", "Phase", "ModelledTime",
                                      "PickTime", "PickError","SNR"])

        p_gauss = np.array([])
        s_gauss = np.array([])
        idx = 0
        for i in range(len(p_onset)):
            p_arrival = max_coa["DT"] + p_ttime[i]
            s_arrival = max_coa["DT"] + s_ttime[i]

            if self.picking_mode == "Gaussian":
                for phase in ["P", "S"]:
                    if phase == "P":
                        onset = p_onset[i]
                        arrival = p_arrival
                    else:
                        onset = s_onset[i]
                        arrival = s_arrival

                    gau, max_onset, err, mn = self._gaussian_picker(onset,
                                                                    phase,
                                                                    start_time,
                                                                    p_arrival,
                                                                    s_arrival,
                                                                    p_ttime[i],
                                                                    s_ttime[i])

                    if phase == "P":
                        p_gauss = np.hstack([p_gauss, gau])
                    else:
                        s_gauss = np.hstack([s_gauss, gau])

                    picks.iloc[idx] = [self.lut.station_data["Name"][i],
                                       phase, arrival, mn, err, max_onset]
                    idx += 1

        self.output.write_picks(picks, event_name)

        return picks, p_gauss, s_gauss

    def _gaufilt3d(self, map_3d, sgm=0.8, shp=None):
        """
        Smooth the 3-D marginalised coalescence map using a 3-D Gaussian
        function

        Parameters
        ----------
        map_3d : 3-d array
            Marginalised 3-d coalescence map

        sgm : float / int
            Sigma value (in grid cells) for the 3d gaussian filter function
            --> bigger sigma leads to more aggressive (long wavelength)
                smoothing.

        shp : array-like, optional
            Shape of volume

        Returns
        -------
        smoothed_map_3d : 3-d array
            Gaussian smoothed 3d coalescence map

        """

        if shp is None:
            shp = map_3d.shape
        nx, ny, nz = shp

        # Normalise
        map_3d = map_3d / np.nanmax(map_3d)

        # Construct 3d gaussian filter
        flt = util.gaussian_3d(nx, ny, nz, sgm, 0.)
        # Convolve map_3d and 3d gaussian filter
        smoothed_map_3d = fftconvolve(map_3d, flt, mode="same")

        # Mirror and convolve again (to avoid "phase-shift")
        smoothed_map_3d = smoothed_map_3d[::-1, ::-1, ::-1] \
            / np.nanmax(smoothed_map_3d)
        smoothed_map_3d = fftconvolve(smoothed_map_3d, flt, mode="same")

        # Final mirror and normalise
        smoothed_map_3d = smoothed_map_3d[::-1, ::-1, ::-1] \
            / np.nanmax(smoothed_map_3d)

        return smoothed_map_3d

    def _mask3d(self, n, i, window):
        """
        Creates a mask that can be applied to a grid

        Parameters
        ----------
        n : array-like, int
            Shape of grid

        i : array-like, int
            Location of cell around which to mask

        window : int
            Size of window around cell to mask

        Returns
        -------
        mask : array-like
            Masking array

        """

        n = np.array(n)
        i = np.array(i)
        w2 = (window - 1) // 2
        x1, y1, z1 = np.clip(i - w2, 0 * n, n)
        x2, y2, z2 = np.clip(i + w2 + 1, 0 * n, n)
        mask = np.zeros(n, dtype=np.bool)
        mask[x1:x2, y1:y2, z1:z2] = True

        return mask

    def _covfit3d(self, coa_map, thresh=0.88, win=None):
        """
        Calculate the 3-D covariance of the marginalised coalescence map

        Parameters
        ----------
        coa_map : 3-d array
            Marginalised 3d coalescence map

        thresh : float (between 0 and 1), optional
            Cut-off threshold (fractional percentile) to trim coa_map; only
            data above this percentile will be retained

        win : int, optional
            Window of grid cells (+/-win in x, y and z) around max value in
            coa_map to perform the fit over

        Returns
        -------
        loc_cov : array-like
            [x, y, z] expectation location from covariance fit

        loc_err_cov : array-like
            [x_err, y_err, z_err] one sigma uncertainties associated with
            loc_cov

        """

        # Normalise
        coa_map = coa_map / (np.nanmax(coa_map))

        # Determining Covariance Location and Error
        nx, ny, nz = coa_map.shape
        mx, my, mz = np.unravel_index(np.nanargmax(coa_map), coa_map.shape)

        # If window is specified, clip the grid to only look here.
        if win:
            flg = np.logical_and(coa_map > thresh,
                                 self._mask3d([nx, ny, nz], [mx, my, mz], win))
            ix, iy, iz = np.where(flg)
            print("Variables", min(ix), max(ix), min(iy), max(iy), min(iz), max(iz))
        else:
            flg = np.where(coa_map > thresh, True, False)
            ix, iy, iz = nx, ny, nz

        smp_weights = coa_map.flatten()
        smp_weights[~flg.flatten()] = np.nan

        lc = self.lut.cell_count
        # Ordering below due to handedness of the grid
        ly, lx, lz = np.meshgrid(np.arange(lc[1]),
                                 np.arange(lc[0]),
                                 np.arange(lc[2]))
        x_samples = lx.flatten() * self.lut.cell_size[0]
        y_samples = ly.flatten() * self.lut.cell_size[1]
        z_samples = lz.flatten() * self.lut.cell_size[2]

        ssw = np.nansum(smp_weights)

        # Expectation values:
        x_expect = np.nansum(smp_weights * x_samples) / ssw
        y_expect = np.nansum(smp_weights * y_samples) / ssw
        z_expect = np.nansum(smp_weights * z_samples) / ssw

        # Covariance matrix:
        cov_matrix = np.zeros((3, 3))
        cov_matrix[0, 0] = np.nansum(smp_weights
                                     * (x_samples - x_expect) ** 2) / ssw
        cov_matrix[1, 1] = np.nansum(smp_weights
                                     * (y_samples - y_expect) ** 2) / ssw
        cov_matrix[2, 2] = np.nansum(smp_weights
                                     * (z_samples - z_expect) ** 2) / ssw
        cov_matrix[0, 1] = np.nansum(smp_weights
                                     * (x_samples - x_expect)
                                     * (y_samples - y_expect)) / ssw
        cov_matrix[1, 0] = cov_matrix[0, 1]
        cov_matrix[0, 2] = np.nansum(smp_weights
                                     * (x_samples - x_expect)
                                     * (z_samples - z_expect)) / ssw
        cov_matrix[2, 0] = cov_matrix[0, 2]
        cov_matrix[1, 2] = np.nansum(smp_weights
                                     * (y_samples - y_expect)
                                     * (z_samples - z_expect)) / ssw
        cov_matrix[2, 1] = cov_matrix[1, 2]

        # Determining the maximum location, and taking 2xgrid cells positive
        # and negative for location in each dimension\

        expect_vector_cov = np.array([x_expect,
                                      y_expect,
                                      z_expect],
                                     dtype=float)
        loc_cov_gc = np.array([[expect_vector_cov[0] / self.lut.cell_size[0],
                                expect_vector_cov[1] / self.lut.cell_size[1],
                                expect_vector_cov[2] / self.lut.cell_size[2]]])

        loc_err_cov = np.array([np.sqrt(cov_matrix[0, 0]),
                                np.sqrt(cov_matrix[1, 1]),
                                np.sqrt(cov_matrix[2, 2])])

        loc_cov = self.lut.xyz2coord(self.lut.xyz2loc(loc_cov_gc,
                                                      inverse=True))[0]

        return loc_cov, loc_err_cov

    def _gaufit3d(self, coa_map, lx=None, ly=None, lz=None, thresh=0., win=7):
        """
        Fit a 3-D Gaussian function to a region around the maximum coalescence
        in the marginalised coalescence map

        Parameters
        ----------
        coa_map : array-like
            Marginalised 3-d coalescence map

        lx : int, optional

        ly : int, optional

        lz : int, optional

        thresh : float (between 0 and 1), optional
            Cut-off threshold (percentile) to trim coa_map: only data above
            this percentile will be retained

        win : int, optional
            Window of grid cells (+/- win in x, y and z) around max value in
            coa_map to perform the fit over

        Returns
        -------
        loc_gau : array-like
            [x, y, z] expectation location from 3-d Gaussian fit

        loc_gau_err : array-like
            [x_err, y_err, z_err] one sigma uncertainties from 3-d Gasussian fit


        """

        nx, ny, nz = coa_map.shape
        mx, my, mz = np.unravel_index(np.nanargmax(coa_map), coa_map.shape)

        # Only use grid cells above threshold value, and within the specified
        # window around the coalescence peak
        flg = np.logical_and(coa_map > thresh,
                             self._mask3d([nx, ny, nz], [mx, my, mz], win))

        ix, iy, iz = np.where(flg)

        # Subtract mean of 3d coalescence map so it is more appropriately
        # approximated by a gaussian (which goes to zero at infinity)
        coa_map = coa_map - np.nanmean(coa_map)

        ncell = len(ix)

        if not lx:
            lx = np.arange(nx)
            ly = np.arange(ny)
            lz = np.arange(nz)

        if lx.ndim == 3:
            iloc = [lx[mx, my, mz], ly[mx, my, mz], lz[mx, my, mz]]
            x = lx[ix, iy, iz] - iloc[0]
            y = ly[ix, iy, iz] - iloc[1]
            z = lz[ix, iy, iz] - iloc[2]
        else:
            iloc = [lx[mx], ly[my], lz[mz]]
            x = lx[ix] - iloc[0]
            y = ly[iy] - iloc[1]
            z = lz[iz] - iloc[2]

        X = np.c_[x * x, y * y, z * z,
                  x * y, x * z, y * z,
                  x, y, z, np.ones(ncell)].T
        Y = -np.log(np.clip(coa_map.astype(np.float64)[ix, iy, iz],
                            1e-300, np.inf))

        X_inv = np.linalg.pinv(X)
        P = np.matmul(Y, X_inv)
        G = -np.array([2 * P[0], P[3], P[4],
                       P[3], 2 * P[1], P[5],
                       P[4], P[5], 2 * P[2]]).reshape((3, 3))
        H = np.array([P[6], P[7], P[8]])
        loc = np.matmul(np.linalg.inv(G), H)
        cx, cy, cz = loc

        K = P[9]             \
            - P[0] * cx ** 2 \
            - P[1] * cy ** 2 \
            - P[2] * cz ** 2 \
            - P[3] * cx * cy \
            - P[4] * cx * cz \
            - P[5] * cy * cz \

        M = np.array([P[0], P[3] / 2, P[4] / 2,
                      P[3] / 2, P[1], P[5] / 2,
                      P[4] / 2, P[5] / 2, P[2]]).reshape(3, 3)
        egv, vec = np.linalg.eig(M)
        sgm = np.sqrt(0.5 / np.clip(np.abs(egv), 1e-10, np.inf))/2
        val = np.exp(-K)
        csgm = np.sqrt(0.5 / np.clip(np.abs(M.diagonal()), 1e-10, np.inf))

        gau_3d = [loc + iloc, vec, sgm, csgm, val]

        # Converting the grid location to X,Y,Z
        xyz = self.lut.xyz2loc(np.array([[gau_3d[0][0],
                                          gau_3d[0][1],
                                          gau_3d[0][2]]]),
                               inverse=True)
        loc_gau = self.lut.xyz2coord(xyz)[0]

        loc_gau_err = np.array([gau_3d[2][0] * self.lut.cell_size[0],
                                gau_3d[2][1] * self.lut.cell_size[1],
                                gau_3d[2][2] * self.lut.cell_size[2]])

        return loc_gau, loc_gau_err

    def _splineloc(self, coa_map, win=5, upscale=10):
        """
        Fit a 3-D spline function to a region around the maximum coalescence
        in the marginalised coalescence map

        Parameters
        ----------
        coa_map : 3-d array
            Marginalised 3-d coalescence map

        win : int
            Window of grid cells (+/- win in x, y and z) around max value in
            coa_map to perform the fit over, optional

        upscale : int
            Upscaling factor to increase the grid ready for spline fitting

        Returns
        -------
        loc : array-like
            [x, y, z] expectation location from spline interpolation

        """

        # np.save("Coamap",coa_map)
        nx, ny, nz = coa_map.shape
        n = np.array([nx, ny, nz])

        mx, my, mz = np.unravel_index(np.nanargmax(coa_map), coa_map.shape)
        i = np.array([mx, my, mz])

        # Determining window about maximum value and trimming coa grid
        w2 = (win - 1)//2
        x1, y1, z1 = np.clip(i - w2, 0 * n, n)
        x2, y2, z2 = np.clip(i + w2 + 1, 0 * n, n)

        # If subgrid is not close to the edge
        if (x2 - x1) == (y2 - y1) == (z2 - z1):
            coa_map_trim = coa_map[x1:x2, y1:y2, z1:z2]

            # Defining the original interpolation function
            xo = np.linspace(0, coa_map_trim.shape[0] - 1, coa_map_trim.shape[0])
            yo = np.linspace(0, coa_map_trim.shape[1] - 1, coa_map_trim.shape[1])
            zo = np.linspace(0, coa_map_trim.shape[2] - 1, coa_map_trim.shape[2])
            xog, yog, zog = np.meshgrid(xo, yo, zo)
            interpgrid = Rbf(xog.flatten(), yog.flatten(), zog.flatten(),
                             coa_map_trim.flatten(),
                             function="cubic")

            # Creating the new grid for the data
            xx = np.linspace(0, coa_map_trim.shape[0] - 1, (coa_map_trim.shape[0] - 1) * upscale + 1)
            yy = np.linspace(0, coa_map_trim.shape[1] - 1, (coa_map_trim.shape[1] - 1) * upscale + 1)
            zz = np.linspace(0, coa_map_trim.shape[2] - 1, (coa_map_trim.shape[2] - 1) * upscale + 1)
            xxg, yyg, zzg = np.meshgrid(xx, yy, zz)
            coa_map_int = interpgrid(xxg.flatten(), yyg.flatten(), zzg.flatten()).reshape(xxg.shape)
            mxi, myi, mzi = np.unravel_index(np.nanargmax(coa_map_int), coa_map_int.shape)
            mxi = mxi/upscale + x1
            myi = myi/upscale + y1
            mzi = mzi/upscale + z1
            print("    \tSpline loc:", mxi, myi, mzi)
            print("    \tGridded loc:", mx, my, mz)

            # Run check that spline location is within grid-cell
            if (abs(mx - mxi) > 1) or (abs(my - myi) > 1) or (abs(mz - mzi) > 1):
                msg = "Spline warning: location outside grid-cell with maximum coalescence value"
                if self.log:
                    self.output.write_log(msg)
                else:
                    print(msg)

            xyz = self.lut.xyz2loc(np.array([[mxi, myi, mzi]]), inverse=True)
            loc = self.lut.xyz2coord(xyz)[0]

            # Run check that spline location is within window
            if (abs(mx - mxi) > w2) or (abs(my - myi) > w2) or (abs(mz - mzi) > w2):
                msg = "Spline error: location outside interpolation window!\n"
                msg += "Gridded Location returned"
                if self.log:
                    self.output.write_log(msg)
                else:
                    print(msg)

                xyz = self.lut.xyz2loc(np.array([[mx, my, mz]]), inverse=True)
                loc = self.lut.xyz2coord(xyz)[0]

        else:
            msg = "Spline error: interpolation window crosses edge of grid!\n"
            msg += "Gridded Location returned"
            if self.log:
                self.output.write_log(msg)
            else:
                print(msg)

            xyz = self.lut.xyz2loc(np.array([[mx, my, mz]]), inverse=True)
            loc = self.lut.xyz2coord(xyz)[0]

        return loc

    def _location_error(self, map_4d):
        """
        Calcuate a set of locations and associated uncertainties using
        the covariance of the coalescence map and by fitting both a 3-D
        Gaussian function and a 3-D spline function to a region around the
        maximum coalescence in the marginalised coalescence map

        Parameters
        ----------
        map_4d : 4-d array
                 4d coalescence grid output from _compute()

        Returns
        -------
        loc : array-like
            [x, y, z] best-fit location from local fit to the coalescence grid

        loc_err : array-like
            [x_err, y_err, z_err] one sigma uncertainties associated with loc

        loc_cov : array-like
            [x, y, z] best-fit location from covariance fit over entire 3d grid
            (most commonly after filtering above a certain percentile).

        loc_err_cov : array-like
            [x_err, y_err, z_err] one sigma uncertainties associated with
            loc_cov

        """

        # MARGINALISE: Determining the coalescence 3D map
        self.coa_map = np.log(np.sum(np.exp(map_4d), axis=-1))

        # Normalise
        self.coa_map = self.coa_map/np.max(self.coa_map)

        # Determining the location error as an error-ellipse
        # Calculate global covariance
        loc_cov, loc_err_cov = self._covfit3d(np.copy(self.coa_map))

        # Fit local gaussian error ellipse
        loc_spline = self._splineloc(np.copy(self.coa_map))
        smoothed_coa_map = self._gaufilt3d(np.copy(self.coa_map))
        loc, loc_err = self._gaufit3d(np.copy(smoothed_coa_map), thresh=0.)

        return loc_spline, loc, loc_err, loc_cov, loc_err_cov

    def _empty(self, w_beg, w_end):
        """
        Create an empty set of arrays to write to .scnmseed

        Parameters
        ----------
        w_beg : UTCDateTime object
            Start datetime to read mSEED

        w_end : UTCDateTime object
            End datetime to read mSEED

        """

        tmp = np.arange(w_beg + self.pre_pad,
                        w_end - self.post_pad + (1 / self.sampling_rate),
                        1 / self.sampling_rate)
        daten = [x.datetime for x in tmp]

        dsnr = dsnr_norm = np.full(len(daten), 0)

        dcoord = np.full((len(daten), 3), 0)

        return daten, dsnr, dsnr_norm, dcoord
