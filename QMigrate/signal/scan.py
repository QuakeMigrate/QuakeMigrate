# -*- coding: utf-8 -*-
"""
Module to perform core QuakeMigrate functions: detect() and locate().

"""

import warnings

import numpy as np
from obspy import UTCDateTime, Stream, Trace
from obspy.signal.invsim import cosine_taper
from obspy.signal.trigger import classic_sta_lta
import pandas as pd
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, fftconvolve

import QMigrate.core.model as qmod
import QMigrate.core.QMigratelib as ilib
import QMigrate.io.quakeio as qio
import QMigrate.plot.quakeplot as qplot
import QMigrate.util as util

# Filter warnings
warnings.filterwarnings("ignore", message=("Covariance of the parameters" +
                                           " could not be estimated"))
warnings.filterwarnings("ignore", message=("File will be written with more" +
                                           " than one different record" +
                                           " lengths. This might have a" +
                                           " negative influence on the" +
                                           " compatibility with other" +
                                           " programs."))


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
        Ratio of short term average to average in a preceding long term average
        window. STA/LTA value is assigned to end of LTA window / start of STA
        window -- "centred"

    """

    nsta = int(round(nsta))
    nlta = int(round(nlta))

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
    Calculate STA/LTA onset (characteristic) function from filtered seismic
    data.

    Parameters
    ----------
    sig : array-like
        Data signal used to generate an onset function

    stw : int
        Short term window length (# of samples)

    ltw : int
        Long term window length (# of samples)

    centred : bool, optional
        Compute centred STA/LTA (STA window is preceded by LTA window; value
        is assigned to end of LTA window / start of STA window) or classic
        STA/LTA (STA window is within LTA window; value is assigned to end of
        STA & LTA windows).

        Centred gives less phase-shifted (late) onset function, and is closer
        to a Gaussian approximation, but is far more sensitive to data with
        sharp offsets due to instrument failures. We recommend using classic
        for detect() and centred for locate() if your data quality allows it.
        This is the default behaviour; override by setting self.onset_centred.

    Returns
    -------
    onset_raw : array-like
        Raw STA/LTA ratio onset function generated from data

    onset : array-like
        log10(onset_raw) ; after clipping between -0.2 and infinity.

    """

    stw = int(round(stw))
    ltw = int(round(ltw))

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


def filter(sig, sampling_rate, lc, hc, order=2):
    """
    Apply zero phase-shift Butterworth band-pass filter to seismic data.

    Parameters
    ----------
    sig : array-like
        Data signal to be filtered

    sampling_rate : int
        Number of samples per second, in Hz

    lc : float
        Lowpass frequency of band-pass filter

    hc : float
        Highpass frequency of band-pass filter

    order : int, optional
        Number of corners. NOTE: two-pass filter effectively doubles the
        number of corners.

    Returns
    -------
    fsig : array-like
        Filtered seismic data

    """

    # Construct butterworth band-pass filter
    b1, a1 = butter(order, [2.0 * lc / sampling_rate,
                            2.0 * hc / sampling_rate], btype="band")
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)

    # Apply cosine taper then apply band-pass filter in both directions
    for ch in range(0, nchan):
        fsig[ch, :] = fsig[ch, :] - fsig[ch, 0]
        tap = cosine_taper(len(fsig[ch, :]), 0.1)
        fsig[ch, :] = fsig[ch, :] * tap
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, ::-1])[::-1]
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, :])

    return fsig


class DefaultQuakeScan(object):
    """
    Default parameter class for QuakeScan.

    """

    def __init__(self):
        """
        Initialise DefaultQuakeScan object.

        Parameters (all optional)
        ----------
        p_bp_filter : array-like, [float, float, int]
            Butterworth bandpass filter specification
            [lowpass, highpass, corners*]
            *NOTE: two-pass filter effectively doubles the number of corners.

        s_bp_filter : array-like, [float, float, int]
            Butterworth bandpass filter specification
            [lowpass, highpass, corners*]
            *NOTE: two-pass filter effectively doubles the number of corners.

        p_onset_win : array-like, floats
            P onset window parameters
            [STA, LTA]

        s_onset_win : array-like, floats
            S onset window parameters
            [STA, LTA]

        decimate : array-like, ints
            Decimation factor in each grid axis to apply to the look-up table
            [Dec_x, Dec_y, Dec_z]

        time_step : float
            Time length (in seconds) of time step used in detect(). Note: total
            detect run duration should be divisible by time_step. Increasing
            time_step will increase RAM usage during detect, but will slightly
            speed up overall detect run.

        sampling_rate : int
            Desired sampling rate for input data; sampling rate that detect()
            and locate() will be computed at.

        onset_centred : bool, optional
            Compute centred STA/LTA (STA window is preceded by LTA window;
            value is assigned to end of LTA window / start of STA window) or
            classic STA/LTA (STA window is within LTA window; value is assigned
            to end of STA & LTA windows).

            Centred gives less phase-shifted (late) onset function, and is
            closer to a Gaussian approximation, but is far more sensitive to
            data with sharp offsets due to instrument failures. We recommend
            using classic for detect() and centred for locate() if your data
            quality allows it. This is the default behaviour; override by
            setting this variable.

        pick_threshold : float (between 0 and 1)
            For use with picking_mode = 'Gaussian'. Picks will only be made if
            the onset function exceeds this percentile of the noise level
            (average amplitude of onset function outside pick windows).
            Recommended starting value: 1.0

        picking_mode : str
            Currently the only available picking mode is 'Gaussian'

        fraction_tt : float
            Defines width of time window around expected phase arrival time in
            which to search for a phase pick as a function of the travel-time
            from the event location to that station -- should be an estimate of
            the uncertainty in the velocity model.

        marginal_window : float
            Time window (+/- marginal_window) about the maximum coalescence
            time to marginalise the 4d coalescence grid compouted in locate()
            to estimate the earthquake location and uncertainty. Should be an
            estimate of the time uncertainty in the earthquake origin time -
            a combination of the expected spatial error and the seismic
            velocity in the region of the event

        pre_pad : float, optional
            Option to override the default pre-pad duration of data to read
            before computing 4d coalescence in detect() and locate(). Default
            value is calculated from the onset function durations.

        n_cores : int
            Number of cores to use on the executing host for detect() /locate()

        continuous_scanmseed_write : bool
            Option to continuously write the .scanmseed file outputted by
            detect() at the end of every time step. Default behaviour is to
            write at the end of the time period, or the end of each day.

        plot_event_summary : bool, optional
            Plot event summary figure - see QMigrate.plot.quakeplot for more
            details.

        plot_station_traces : bool, optional
            Plot data and onset functions overlain by phase picks for each
            station used in locate()

        plot_coal_video : bool, optional
            Plot coalescence video for each earthquake located in locate()

        write_4d_coal_grid : bool, optional
            Save the full 4d coalescence grid output by compute for each event
            located by locate() -- NOTE these files are large.

        write_cut_waveforms : bool, optional
            Write raw cut waveforms for all data found in the archive for each
            event located by locate() -- NOTE this data has not been processed
            or quality-checked!

        cut_waveform_format : str, optional
            File format to write waveform data to. Options are all file formats
            supported by obspy, including: "MSEED" (default), "SAC", "SEGY",
            "GSE2"

        pre_cut : float, optional
            Specify how long before the event origin time to cut the waveform
            data from

        post_cut : float, optional
            Specify how long after the event origin time to cut the waveform
            data to

        xy_files : list, string
            List of file strings:
            With columns ["File", "Color", "Linewidth", "Linestyle"]
            Where File is the file path to the xy file to be plotted on the
            map. File should contain two columns ["Longitude", "Latitude"].
            ** NOTE ** - do not include a header line in either file.

        """

        # Filter parameters
        self.p_bp_filter = [2.0, 16.0, 2]
        self.s_bp_filter = [2.0, 12.0, 2]

        # Onset window parameters
        self.p_onset_win = [0.2, 1.0]
        self.s_onset_win = [0.2, 1.0]

        # Traveltime lookup table decimation factor
        self.decimate = [1, 1, 1]

        # Time step for continuous compute in detect
        self.time_step = 120.

        # Data sampling rate
        self.sampling_rate = 50

        # Centred onset function override -- None means it will be
        # automatically set in detect() and locate()
        self.onset_centred = None

        # Pick related parameters
        self.pick_threshold = 1.0
        self.picking_mode = "Gaussian"
        self.fraction_tt = 0.1

        # Marginal window
        self.marginal_window = 2.

        # Default pre-pad for compute
        self.pre_pad = None

        # Number of cores to perform detect/locate on
        self.n_cores = 1

        # Toggle whether to incrementally write .scanmseed in detect()
        self.continuous_scanmseed_write = False

        # Plotting toggles
        self.plot_event_summary = True
        self.plot_station_traces = False
        self.plot_coal_video = False

        # Saving toggles
        self.write_4d_coal_grid = False
        self.write_cut_waveforms = False
        self.cut_waveform_format = "MSEED"
        self.pre_cut = None
        self.post_cut = None

        # xy files for plotting
        self.xy_files = None


class QuakeScan(DefaultQuakeScan):
    """
    QuakeMigrate scanning class

    Forms the core of the QuakeMigrate method, providing wrapper functions for
    the C-compiled migration methods.

    Methods
    -------
    detect(start_time, end_time)
        Core detection method -- compute decimated 3-D coalescence continuously
                                 throughout entire time period; output as
                                 .scanmseed (in mSEED format).

    locate(start_time, end_time)
        Core locate method -- compute 3-D coalescence over short time window
                              around candidate earthquake triggered from
                              coastream; output location & uncertainties
                              (.event file), phase picks (.picks file), plus
                              multiple optional plots / data for further
                              analysis and processing.

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

    def __init__(self, data, lookup_table, output_path=None, run_name=None, log=False):
        """
        Class initialisation method.

        Parameters
        ----------
        data : Archive object
            Contains information on data archive structure and
            read_waveform_data() method

        lookup_table : str
            Look-up table file path

        output_path : str
            Path of parent output directory: e.g. ./OUTPUT

        run_name : str
            Name of current run: all outputs will be saved in the directory
            output_path/run_name

        """

        DefaultQuakeScan.__init__(self)

        self.data = data
        lut = qmod.LUT()
        lut.load(lookup_table)
        self.lut = lut

        if output_path is not None:
            self.output = qio.QuakeIO(output_path, run_name, log)
        else:
            self.output = None

        # Define post-pad as a function of the maximum travel-time between a
        # station and a grid point plus the LTA (in case onset_centred is True)
        #  ---> applies to both detect() and locate()
        ttmax = np.max(lut.fetch_map("TIME_S"))
        lta_max = max(self.p_onset_win[1], self.s_onset_win[1])
        self.post_pad = np.ceil(ttmax + 2 * lta_max)

        self.log = log
        msg = "=" * 120 + "\n"
        msg += "=" * 120 + "\n"
        msg += "\tQuakeMigrate - Coalescence Scanning - Path: {} - Name: {}\n"
        msg += "=" * 120 + "\n"
        msg += "=" * 120 + "\n"
        msg = msg.format(self.output.path, self.output.name)
        self.output.log(msg, self.log)

    def __str__(self):
        """
        Return short summary string of the QuakeScan object

        It will provide information on all of the various parameters that the
        user can/has set.

        """

        out = "QuakeMigrate parameters"
        out += "\n\tTime step\t\t:\t{}".format(self.time_step)
        out += "\n\n\tData sampling rate\t:\t{}".format(self.sampling_rate)
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
        out += "\n\tFraction ttime\t\t:\t{}".format(self.fraction_tt)
        out += "\n\n\tCentred onset\t\t:\t{}".format(self.onset_centred)
        out += "\n\n\tNumber of CPUs\t\t:\t{}".format(self.n_cores)

        return out

    def detect(self, start_time, end_time):
        """
        Scans through continuous data calculating coalescence on a decimated
        3D grid by back-migrating P and S onset (characteristic) functions.

        Parameters
        ----------
        start_time : str
            Start time of continuous scan

        end_time : str
            End time of continuous scan (last sample returned will be that
            which immediately precedes this time stamp)

        log : bool, optional
            Write output to a log file (default: False)

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        # Decimate LUT
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

        msg = "=" * 120 + "\n"
        msg += "\tDETECT - Continuous Seismic Processing\n"
        msg += "=" * 120 + "\n"
        msg += "\n"
        msg += "\tParameters specified:\n"
        msg += "\t\tStart time                = {}\n"
        msg += "\t\tEnd   time                = {}\n"
        msg += "\t\tTime step (s)             = {}\n"
        msg += "\t\tNumber of CPUs            = {}\n"
        msg += "\n"
        msg += "\t\tSampling rate             = {}\n"
        msg += "\t\tGrid decimation [X, Y, Z] = [{}, {}, {}]\n"
        msg += "\t\tBandpass filter P         = [{}, {}, {}]\n"
        msg += "\t\tBandpass filter S         = [{}, {}, {}]\n"
        msg += "\t\tOnset P [STA, LTA]        = [{}, {}]\n"
        msg += "\t\tOnset S [STA, LTA]        = [{}, {}]\n"
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
        self.output.log(msg, self.log)

        # Detect max coalescence value and location at each time sample
        # within the decimated grid
        self._continuous_compute(start_time, end_time)

    def locate(self, start_time, end_time):
        """
        Re-computes the 3D coalescence on a less decimated grid for a short
        time window around each candidate earthquake triggered from the
        decimated continuous detect scan. Calculates event location and
        uncertainties, makes phase arrival picks, plus multiple optional
        plotting / data outputs for further analysis and processing.

        Parameters
        ----------
        start_time : str
            Start time of locate run: earliest event trigger time that will be
            located

        end_time : str
            End time of locate run: latest event trigger time that will be
            located is one sample before this time

        log : bool, optional
            Write output to a log file (default: False)

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        msg = "=" * 120 + "\n"
        msg += "\tLOCATE - Determining earthquake location and uncertainty\n"
        msg += "=" * 120 + "\n"
        msg += "\n"
        msg += "\tParameters specified:\n"
        msg += "\t\tStart time                = {}\n"
        msg += "\t\tEnd   time                = {}\n"
        msg += "\t\tNumber of CPUs            = {}\n\n"
        msg += "=" * 120 + "\n"
        msg = msg.format(str(start_time), str(end_time), self.n_cores)
        self.output.log(msg, self.log)

        # Decimate LUT
        self.lut = self.lut.decimate(self.decimate)

        # Locate uses the centred onset by default
        if self.onset_centred is None:
            self.onset_centred = True

        self._locate_events(start_time, end_time)

    def _append_coastream(self, coastream, daten, max_coa, max_coa_norm, loc,
                          sampling_rate):
        """
        Append latest timestep of detect() output to obspy.Stream() object.
        Multiply by factor of ["1e5", "1e5", "1e6", "1e6", "1e3"] respectively
        for channels ["COA", "COA_N", "X", "Y", "Z"], round and convert to
        int32 as this dramatically reduces memory usage, and allows the
        coastream data to be saved in mSEED format with STEIM2 compression.
        The multiplication factor is removed when the data is read back in.

        Parameters
        ----------
        coastream : obspy Stream object
            Data output by detect() so far
            channels: ["COA", "COA_N", "X", "Y", "Z"]
            NOTE these values have been multiplied by a factor and converted to
            an int

        daten : array-like
            Array of UTCDateTime time stamps for the time step

        max_coa : array-like
            Coalescence value through time

        max_coa_norm : array-like
            Normalised coalescence value through time

        loc : array-like
            Location of maximum coalescence through time

        sampling_rate : int
            Sampling rate that detect is run at.

        Returns
        -------
        coastream : obspy Stream object
            Data output by detect() so far with most recent timestep appended
            channels: ["COA", "COA_N", "X", "Y", "Z"]
            NOTE these values have been multiplied by a factor and converted to
            an int

        """

        # clip max value of COA to prevent int overflow
        max_coa[max_coa > 21474.] = 21474.
        max_coa_norm[max_coa_norm > 21474.] = 21474.

        npts = len(max_coa)
        starttime = UTCDateTime(daten[0])
        meta = {"network": "NW",
                "npts": npts,
                "sampling_rate": sampling_rate,
                "starttime": starttime}

        st = Stream(Trace(data=np.round(max_coa * 1e5).astype(np.int32),
                          header={**{"station": "COA"}, **meta}))
        st += Stream(Trace(data=np.round(max_coa_norm * 1e5).astype(np.int32),
                           header={**{"station": "COA_N"}, **meta}))
        st += Stream(Trace(data=np.round(loc[:, 0] * 1e6).astype(np.int32),
                           header={**{"station": "X"}, **meta}))
        st += Stream(Trace(data=np.round(loc[:, 1] * 1e6).astype(np.int32),
                           header={**{"station": "Y"}, **meta}))
        st += Stream(Trace(data=np.round(loc[:, 2] * 1e3).astype(np.int32),
                           header={**{"station": "Z"}, **meta}))

        if coastream is not None:
            coastream = coastream + st
            coastream.merge(method=-1)
        else:
            coastream = st

        # Have we moved to the next day? If so write out the file and empty
        # coastream
        written = False
        if coastream[0].stats.starttime.julday != \
           coastream[0].stats.endtime.julday:
            write_start = coastream[0].stats.starttime
            write_end = UTCDateTime(coastream[0].stats.endtime.date) \
                        - 1 / coastream[0].stats.sampling_rate

            self.output.write_coastream(coastream, write_start, write_end)
            written = True

            coastream.trim(starttime=write_end + 1 / sampling_rate)

        return coastream, written

    def _continuous_compute(self, start_time, end_time):
        """
        Compute coalescence between two time stamps, divided into small time
        steps. Outputs coastream and station availability data to file.

        Parameters
        ----------
        start_time : UTCDateTime object
            Time stamp of first sample

        end_time : UTCDateTime object
            Time stamp of final sample

        """

        coastream = None

        t_length = self.pre_pad + self.post_pad + self.time_step
        self.pre_pad += np.ceil(t_length * 0.06)
        self.post_pad += np.ceil(t_length * 0.06)

        try:
            nsteps = int(np.ceil((end_time - start_time) / self.time_step))
        except AttributeError:
            msg = "Error: Time step has not been specified"
            self.output.log(msg, self.log)

        # Initialise pandas DataFrame object to track availability
        stn_ava_data = pd.DataFrame(index=np.arange(nsteps),
                                    columns=self.data.stations)

        for i in range(nsteps):
            timer = util.Stopwatch()
            w_beg = start_time + self.time_step * i - self.pre_pad
            w_end = start_time + self.time_step * (i + 1) + self.post_pad

            msg = ("~" * 24) + " Processing : {} - {} " + ("~" * 24)
            msg = msg.format(str(w_beg), str(w_end))
            self.output.log(msg, self.log)

            try:
                self.data.read_waveform_data(w_beg, w_end, self.sampling_rate)
                daten, max_coa, max_coa_norm, loc, map_4d = self._compute(
                                                          w_beg, w_end,
                                                          self.data.signal,
                                                          self.data.availability)
                stn_ava_data.loc[i] = self.data.availability
                coord = self.lut.xyz2coord(loc)

                del loc, map_4d

            except util.ArchiveEmptyException:
                msg = "!" * 24 + " " * 16
                msg += " No files in archive for this time step "
                msg += " " * 16 + "!" * 24
                self.output.log(msg, self.log)
                daten, max_coa, max_coa_norm, coord = self._empty(w_beg, w_end)
                stn_ava_data.loc[i] = self.data.availability

            except util.DataGapException:
                msg = "!" * 24 + " " * 9
                msg += "All available data for this time period contains gaps"
                msg += " " * 10 + "!" * 24
                msg += "\n" + "!" * 24 + " " * 11
                msg += "or data not available at start/end of time period"
                msg += " " * 12 + "!" * 24
                self.output.log(msg, self.log)
                daten, max_coa, max_coa_norm, coord = self._empty(w_beg, w_end)
                stn_ava_data.loc[i] = self.data.availability

            stn_ava_data.rename(index={i: str(w_beg + self.pre_pad)},
                                inplace=True)

            # Append upto sample-before-last - if end_time is
            # 2014-08-24T00:00:00, your last sample will be 2014-08-23T23:59:59
            coastream, written = self._append_coastream(coastream,
                                                        daten[:-1],
                                                        max_coa[:-1],
                                                        max_coa_norm[:-1],
                                                        coord[:-1, :],
                                                        self.sampling_rate)

            del daten, max_coa, max_coa_norm, coord

            if self.continuous_scanmseed_write and not written:
                self.output.write_coastream(coastream)
                written = True

            self.output.log(timer(), self.log)

        if not written:
            self.output.write_coastream(coastream)

        del coastream

        self.output.write_stn_availability(stn_ava_data)

        self.output.log("=" * 120, self.log)

    def _locate_events(self, start_time, end_time):
        """
        Loop through list of earthquakes read in from trigger results and
        re-compute coalescence; output phase picks, event location and
        uncertainty, plus optional plots and outputs.

        Parameters
        ----------
        start_time : UTCDateTime object
           Start time of locate run: earliest event trigger time that will be
            located

        end_time : UTCDateTime object
            End time of locate run: latest event trigger time that will be
            located

        """

        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.p_onset_win[1],
                               self.s_onset_win[1]) \
                           + 3 * max(self.p_onset_win[0],
                                     self.s_onset_win[0])

        # Adjust pre- and post-pad to take into account cosine taper
        t_length = self.pre_pad + 4*self.marginal_window + self.post_pad
        self.pre_pad += np.ceil(t_length * 0.06)
        self.post_pad += np.ceil(t_length * 0.06)

        trig_events = self.output.read_triggered_events(start_time, end_time)
        n_evts = len(trig_events)

        for i, trig_event in trig_events.iterrows():
            event_uid = trig_event["EventID"]
            msg = "=" * 120 + "\n"
            msg += "\tEVENT - {} of {} - {}\n"
            msg += "=" * 120 + "\n\n"
            msg += "\tDetermining event location...\n"
            msg = msg.format(i + 1, n_evts, event_uid)
            self.output.log(msg, self.log)

            w_beg = trig_event["CoaTime"] - 2*self.marginal_window \
                - self.pre_pad
            w_end = trig_event["CoaTime"] + 2*self.marginal_window \
                + self.post_pad

            timer = util.Stopwatch()
            self.output.log("\tReading waveform data...", self.log)
            try:
                self._read_event_waveform_data(trig_event, w_beg, w_end)
            except util.ArchiveEmptyException:
                msg = "\tNo files found in archive for this time period"
                self.output.log(msg, self.log)
                continue
            except util.DataGapException:
                msg = "\tAll available data for this time period contains gaps"
                msg += "\n\tOR data not available at start/end of time period\n"
                self.output.log(msg, self.log)
                continue
            self.output.log(timer(), self.log)

            timer = util.Stopwatch()
            self.output.log("\tComputing 4D coalescence grid...", self.log)

            daten, max_coa, max_coa_norm, loc, map_4d = self._compute(
                                                      w_beg, w_end,
                                                      self.data.signal,
                                                      self.data.availability)
            coord = self.lut.xyz2coord(np.array(loc).astype(int))
            event_coa_data = pd.DataFrame(np.array((daten, max_coa,
                                                    coord[:, 0],
                                                    coord[:, 1],
                                                    coord[:, 2])).transpose(),
                                          columns=["DT", "COA", "X", "Y", "Z"])
            event_coa_data["DT"] = event_coa_data["DT"].apply(UTCDateTime)
            event_coa_data_dtmax = \
                event_coa_data["DT"].iloc[event_coa_data["COA"].astype("float").idxmax()]
            w_beg_mw = event_coa_data_dtmax - self.marginal_window
            w_end_mw = event_coa_data_dtmax + self.marginal_window

            if (event_coa_data_dtmax >= trig_event["CoaTime"]
                - self.marginal_window) \
               and (event_coa_data_dtmax <= trig_event["CoaTime"]
                    + self.marginal_window):
                w_beg_mw = event_coa_data_dtmax - self.marginal_window
                w_end_mw = event_coa_data_dtmax + self.marginal_window
            else:
                msg = "\n\tEvent {} is outside marginal window.\n"
                msg += "\tDefine more realistic error - the marginal window"
                msg += " should be an estimate of the origin time uncertainty,"
                msg += "\n\tdetermined by the expected spatial uncertainty and"
                msg += "the seismic velocity in the region of the earthquake\n"
                msg += "\n" + "=" * 120 + "\n"
                msg = msg.format(event_uid)
                self.output.log(msg, self.log)
                continue

            event_mw_data = event_coa_data
            event_mw_data = event_mw_data[(event_mw_data["DT"] >= w_beg_mw) &
                                          (event_mw_data["DT"] <= w_end_mw)]
            map_4d = map_4d[:, :, :,
                            event_mw_data.index[0]:event_mw_data.index[-1]]
            event_mw_data = event_mw_data.reset_index(drop=True)
            event_max_coa = event_mw_data.iloc[event_mw_data["COA"].astype("float").idxmax()]

            # Update event UID; make out_str
            event_uid = str(event_max_coa.values[0])
            for char_ in ["-", ":", ".", " ", "Z", "T"]:
                event_uid = event_uid.replace(char_, "")
            out_str = "{}_{}".format(self.output.name, event_uid)
            self.output.log(timer(), self.log)

            # Make phase picks
            timer = util.Stopwatch()
            self.output.log("\tMaking phase picks...", self.log)
            phase_picks = self._phase_picker(event_max_coa)
            self.output.write_picks(phase_picks["Pick"], event_uid)
            self.output.log(timer(), self.log)

            # Determining earthquake location error
            timer = util.Stopwatch()
            self.output.log("\tDetermining earthquake location and uncertainty...", self.log)
            loc_spline, loc_gau, loc_gau_err, loc_cov, \
                loc_cov_err = self._calculate_location(map_4d)
            self.output.log(timer(), self.log)

            # Make event dictionary with all final event location data
            event = pd.DataFrame([[event_max_coa.values[0],
                                   event_max_coa.values[1],
                                   loc_spline[0], loc_spline[1], loc_spline[2],
                                   loc_gau[0], loc_gau[1], loc_gau[2],
                                   loc_gau_err[0], loc_gau_err[1],
                                   loc_gau_err[2],
                                   loc_cov[0], loc_cov[1], loc_cov[2],
                                   loc_cov_err[0], loc_cov_err[1],
                                   loc_cov_err[2]]],
                                 columns=self.EVENT_FILE_COLS)

            self.output.write_event(event, event_uid)

            self._optional_locate_outputs(event_mw_data, event, out_str,
                                          phase_picks, event_uid, map_4d)

            self.output.log("=" * 120 + "\n", self.log)

            del map_4d, event_coa_data, event_mw_data, event_max_coa, \
                phase_picks
            self.coa_map = None

    def _read_event_waveform_data(self, event, w_beg, w_end):
        """
        Read waveform data for a triggered event.

        Parameters
        ----------
        event : pandas DataFrame
            Triggered event output from _trigger_scn().
            Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                      "MinTime", "MaxTime"]

        w_beg : UTCDateTime object
            Start datetime to read waveform data

        w_end : UTCDateTime object
            End datetime to read waveform data

        Returns
        -------
        daten, max_coa, max_coa_norm, coord : array-like
            Empty arrays with the correct shape to write to .scanmseed as if
            they were coastream outputs from _compute()

        """

        # Extra pre- and post-pad default to None
        pre_pad = post_pad = None
        if self.pre_cut:
            # only subtract 1*marginal_window so if the event otime moves by
            # this much the selected pre_cut can still be applied
            pre_pad = self.pre_cut - self.marginal_window - self.pre_pad
            if pre_pad < 0:
                msg = "\t\tWarning: specified pre_cut {} is shorter than"
                msg += "default pre_pad\n"
                msg += "\t\t\tCutting from pre_pad = {}"
                msg = msg.format(self.pre_cut, self.pre_pad)
                self.output.log(msg, self.log)

                pre_pad = None

        if self.post_cut:
            # only subtract 1*marginal_window so if the event otime moves by
            # this much the selected post_cut can still be applied
            post_pad = self.post_cut - self.marginal_window - \
                       self.post_pad
            if post_pad < 0:
                msg = "\t\tWarning: specified post_cut {} is shorter than"
                msg += "default post_pad\n"
                msg += "\t\t\tCutting to post_pad = {}"
                msg = msg.format(self.post_cut, self.post_pad)
                self.output.log(msg, self.log)
                post_pad = None

        self.data.read_waveform_data(w_beg, w_end, self.sampling_rate, pre_pad,
                                     post_pad)

    def _compute(self, w_beg, w_end, signal, station_availability):
        """
        Compute 3-D coalescence between two time stamps.

        Parameters
        ----------
        w_beg : UTCDateTime object
            Time stamp of first sample in window

        w_end : UTCDateTime object
            Time stamp of final sample in window

        signal : array-like
            Pre-processed continuous 3-component data stream for all available
            stations -- linearly detrended, de-meaned, resampled if necessary

        station_availability : array-like
            List of available stations

        Returns
        -------
        daten : array-like
            UTCDateTime time stamp for each sample between w_beg and w_end

        max_coa : array-like
            Coalescence value through time

        max_coa_norm : array-like
            Normalised coalescence value through time

        loc : array-like
            Location of maximum coalescence through time

        map_4d : array-like
            4-D coalescence map

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

        # Prep empty 4-D coalescence map and run C-compiled ilib.migrate()
        ncell = tuple(self.lut.cell_count)
        map_4d = np.zeros(ncell + (nsamp,), dtype=np.float64)
        ilib.migrate(ps_onset, ttime, pre_smp, pos_smp, nsamp, map_4d,
                     self.n_cores)

        # Prep empty coa and loc arrays and run C-compiled ilib.find_max_coa()
        max_coa = np.zeros(nsamp, np.double)
        grid_index = np.zeros(nsamp, np.int64)
        ilib.find_max_coa(map_4d, max_coa, grid_index, 0, nsamp, self.n_cores)

        # Get max_coa_norm
        sum_coa = np.sum(map_4d, axis=(0, 1, 2))
        max_coa_norm = max_coa / sum_coa
        max_coa_norm = max_coa_norm * map_4d.shape[0] * map_4d.shape[1] * \
                       map_4d.shape[2]

        tmp = np.arange(w_beg + self.pre_pad,
                        w_end - self.post_pad + (1 / self.sampling_rate),
                        1 / self.sampling_rate)
        daten = [x.datetime for x in tmp]

        # Calculate max_coa (with correction for number of stations)
        max_coa = np.exp((max_coa / (len(avail_idx) * 2)) - 1.0)

        loc = self.lut.xyz2index(grid_index, inverse=True)

        return daten, max_coa, max_coa_norm, loc, map_4d

    def _compute_p_onset(self, sig_z, sampling_rate):
        """
        Generates an onset (characteristic) function for the P-phase from the
        Z-component signal.

        Parameters
        ----------
        sig_z : array-like
            Z-component time-series data

        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        p_onset_raw : array-like
            Onset function generated from raw STA/LTA of vertical component
            data

        p_onset : array-like
            Onset function generated from log10(STA/LTA) of vertical
            component data

        """

        stw, ltw = self.p_onset_win
        stw = int(stw * sampling_rate) + 1
        ltw = int(ltw * sampling_rate) + 1

        lc, hc, ord_ = self.p_bp_filter
        filt_sig_z = filter(sig_z, sampling_rate, lc, hc, ord_)
        self.data.filtered_signal[2, :, :] = filt_sig_z

        p_onset_raw, p_onset = onset(filt_sig_z, stw, ltw,
                                     centred=self.onset_centred)
        self.onset_data["sigz"] = p_onset

        return p_onset_raw, p_onset

    def _compute_s_onset(self, sig_e, sig_n, sampling_rate):
        """
        Generates an onset (characteristic) function for the S-phase from the
        E- and N-components signal.

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
            Onset function generated from raw STA/LTA of horizontal component
            data

        s_onset : array-like
            Onset function generated from log10(STA/LTA) of horizontal
            component data

        """

        stw, ltw = self.s_onset_win
        stw = int(stw * sampling_rate) + 1
        ltw = int(ltw * sampling_rate) + 1

        lc, hc, ord_ = self.s_bp_filter
        filt_sig_e = filter(sig_e, sampling_rate, lc, hc, ord_)
        filt_sig_n = filter(sig_n, sampling_rate, lc, hc, ord_)
        self.data.filtered_signal[0, :, :] = filt_sig_n
        self.data.filtered_signal[1, :, :] = filt_sig_e

        s_e_onset_raw, s_e_onset = onset(filt_sig_e, stw, ltw,
                                         centred=self.onset_centred)
        s_n_onset_raw, s_n_onset = onset(filt_sig_n, stw, ltw,
                                         centred=self.onset_centred)
        self.onset_data["sige"] = s_e_onset
        self.onset_data["sign"] = s_n_onset

        s_onset = np.sqrt((s_e_onset ** 2 + s_n_onset ** 2) / 2.)
        s_onset_raw = np.sqrt((s_e_onset_raw ** 2 + s_n_onset_raw ** 2) / 2.)
        self.onset_data["sigs"] = s_onset

        return s_onset_raw, s_onset

    def _gaussian_picker(self, onset, phase, start_time, p_arr, s_arr, ptt,
                         stt):
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

    def _phase_picker(self, event):
        """
        Picks phase arrival times for located earthquakes.

        Parameters
        ----------
        event : pandas DataFrame
            Contains data about located event.
            Columns: ["DT", "COA", "X", "Y", "Z"] - X and Y as lon/lat; Z in m

        Returns
        -------
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

        """

        event_crd = np.array([event[["X", "Y", "Z"]].values])
        event_xyz = np.array(self.lut.xyz2coord(event_crd,
                                                inverse=True)).astype(int)[0]

        p_ttime = self.lut.value_at("TIME_P", event_xyz)[0]
        s_ttime = self.lut.value_at("TIME_S", event_xyz)[0]

        # Determining the stations that can be picked on and the phases
        picks = pd.DataFrame(index=np.arange(0, 2 * len(self.data.p_onset)),
                             columns=["Name", "Phase", "ModelledTime",
                                      "PickTime", "PickError", "SNR"])

        p_gauss = np.array([])
        s_gauss = np.array([])
        idx = 0
        for i in range(len(self.data.p_onset)):
            p_arrival = event["DT"] + p_ttime[i]
            s_arrival = event["DT"] + s_ttime[i]

            if self.picking_mode == "Gaussian":
                for phase in ["P", "S"]:
                    if phase == "P":
                        onset = self.data.p_onset[i]
                        arrival = p_arrival
                    else:
                        onset = self.data.s_onset[i]
                        arrival = s_arrival

                    gau, max_onset, err, mn = \
                                    self._gaussian_picker(onset,
                                                          phase,
                                                          self.data.start_time,
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

        phase_picks = {}
        phase_picks["Pick"] = picks
        phase_picks["GAU_P"] = p_gauss
        phase_picks["GAU_S"] = s_gauss

        return phase_picks

    def _gaufilt3d(self, map_3d, sgm=0.8, shp=None):
        """
        Smooth the 3-D marginalised coalescence map using a 3-D Gaussian
        function to enable a better gaussian fit to the data to be calculated.

        Parameters
        ----------
        map_3d : array-like
            Marginalised 3-D coalescence map

        sgm : float
            Sigma value (in grid cells) for the 3d gaussian filter function;
            bigger sigma leads to more aggressive (long wavelength) smoothing

        shp : array-like, optional
            Shape of volume

        Returns
        -------
        smoothed_map_3d : array-like
            Gaussian smoothed 3-D coalescence map

        """

        if shp is None:
            shp = map_3d.shape
        nx, ny, nz = shp

        # Normalise
        map_3d = map_3d / np.nanmax(map_3d)

        # Construct 3d gaussian filter
        flt = util.gaussian_3d(nx, ny, nz, sgm)

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
        Creates a mask that can be applied to a 3-D grid.

        Parameters
        ----------
        n : array-like, int
            Shape of grid

        i : array-like, int
            Location of cell around which to mask

        window : int
            Size of window around cell to mask - window of grid cells is
            +/-(win-1)//2 in x, y and z

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
        Calculate the 3-D covariance of the marginalised coalescence map,
        filtered above a percentile threshold {thresh}. Optionally can also
        perform the fit on a sub-window of the grid around the maximum
        coalescence location.

        Parameters
        ----------
        coa_map : array-like
            Marginalised 3-D coalescence map

        thresh : float (between 0 and 1), optional
            Cut-off threshold (fractional percentile) to trim coa_map; only
            data above this percentile will be retained

        win : int, optional
            Window of grid cells (+/-(win-1)//2 in x, y and z) around max
            value in coa_map to perform the fit over

        Returns
        -------
        loc_cov : array-like
            [x, y, z] : expectation location from covariance fit

        loc_cov_err : array-like
            [x_err, y_err, z_err] : one sigma uncertainties associated with
                                    loc_cov

        """

        # Normalise
        coa_map = coa_map / (np.nanmax(coa_map))

        # Get shape of 3-D coalescence map and max coalesence grid location
        nx, ny, nz = coa_map.shape
        mx, my, mz = np.unravel_index(np.nanargmax(coa_map), coa_map.shape)

        # If window is specified, clip the grid to only look here.
        if win:
            flg = np.logical_and(coa_map > thresh,
                                 self._mask3d([nx, ny, nz], [mx, my, mz], win))
            ix, iy, iz = np.where(flg)
            msg = "Variables", min(ix), max(ix), min(iy), max(iy), min(iz), max(iz)
            self.output.log(msg, self.log)
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

        expect_vector_cov = np.array([x_expect,
                                      y_expect,
                                      z_expect],
                                     dtype=float)
        loc_cov_gc = np.array([[expect_vector_cov[0] / self.lut.cell_size[0],
                                expect_vector_cov[1] / self.lut.cell_size[1],
                                expect_vector_cov[2] / self.lut.cell_size[2]]])
        loc_cov_err = np.array([np.sqrt(cov_matrix[0, 0]),
                                np.sqrt(cov_matrix[1, 1]),
                                np.sqrt(cov_matrix[2, 2])])

        # Convert grid location to XYZ / coordinates
        xyz = self.lut.xyz2loc(loc_cov_gc, inverse=True)
        loc_cov = self.lut.xyz2coord(xyz)[0]

        return loc_cov, loc_cov_err

    def _gaufit3d(self, coa_map, lx=None, ly=None, lz=None, thresh=0., win=7):
        """
        Fit a 3-D Gaussian function to a region around the maximum coalescence
        location in the 3-D marginalised coalescence map: return expectation
        location and associated uncertainty.

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
            Window of grid cells (+/-(win-1)//2 in x, y and z) around max
            value in coa_map to perform the fit over

        Returns
        -------
        loc_gau : array-like
            [x, y, z] : expectation location from 3-d Gaussian fit

        loc_gau_err : array-like
            [x_err, y_err, z_err] : one sigma uncertainties from 3-d Gaussian
                                    fit

        """

        # Get shape of 3-D coalescence map and max coalesence grid location
        nx, ny, nz = coa_map.shape
        mx, my, mz = np.unravel_index(np.nanargmax(coa_map), coa_map.shape)

        # Only use grid cells above threshold value, and within the specified
        # window around the coalescence peak
        flg = np.logical_and(coa_map > thresh,
                             self._mask3d([nx, ny, nz], [mx, my, mz], win))
        ix, iy, iz = np.where(flg)

        # Subtract mean of entire 3-D coalescence map from the local grid
        # window so it is better approximated by a gaussian (which goes to zero
        # at infinity)
        coa_map = coa_map - np.nanmean(coa_map)

        # Fit 3-D gaussian function
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

        # Convert back to whole-grid coordinates
        gau_3d = [loc + iloc, vec, sgm, csgm, val]

        # Convert grid location to XYZ / coordinates
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
        in the marginalised coalescence map and interpolate by factor {upscale}
        to return a sub-grid maximum coalescence location.

        Parameters
        ----------
        coa_map : array-like
            Marginalised 3-D coalescence map

        win : int
            Window of grid cells (+/-(win-1)//2 in x, y and z) around max
            value in coa_map to perform the fit over

        upscale : int
            Upscaling factor to interpolate the fitted 3-D spline function by

        Returns
        -------
        loc_spline : array-like
            [x, y, z] : max coalescence location from spline interpolation

        """

        # Get shape of 3-D coalescence map
        nx, ny, nz = coa_map.shape
        n = np.array([nx, ny, nz])

        # Find maximum coalescence location in grid
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
            xo = np.linspace(0, coa_map_trim.shape[0] - 1,
                             coa_map_trim.shape[0])
            yo = np.linspace(0, coa_map_trim.shape[1] - 1,
                             coa_map_trim.shape[1])
            zo = np.linspace(0, coa_map_trim.shape[2] - 1,
                             coa_map_trim.shape[2])
            xog, yog, zog = np.meshgrid(xo, yo, zo)
            interpgrid = Rbf(xog.flatten(), yog.flatten(), zog.flatten(),
                             coa_map_trim.flatten(),
                             function="cubic")

            # Creating the new interpolated grid
            xx = np.linspace(0, coa_map_trim.shape[0] - 1,
                             (coa_map_trim.shape[0] - 1) * upscale + 1)
            yy = np.linspace(0, coa_map_trim.shape[1] - 1,
                             (coa_map_trim.shape[1] - 1) * upscale + 1)
            zz = np.linspace(0, coa_map_trim.shape[2] - 1,
                             (coa_map_trim.shape[2] - 1) * upscale + 1)
            xxg, yyg, zzg = np.meshgrid(xx, yy, zz)

            # Interpolate spline function on new grid
            coa_map_int = interpgrid(xxg.flatten(), yyg.flatten(),
                                     zzg.flatten()).reshape(xxg.shape)

            # Calculate max coalescence location on interpolated grid
            mxi, myi, mzi = np.unravel_index(np.nanargmax(coa_map_int),
                                             coa_map_int.shape)
            mxi = mxi/upscale + x1
            myi = myi/upscale + y1
            mzi = mzi/upscale + z1
            self.output.log("\t\tGridded loc: {}   {}   {}".format(mx, my, mz), self.log)
            self.output.log("\t\tSpline  loc: {} {} {}".format(mxi, myi, mzi), self.log)

            # Run check that spline location is within grid-cell
            if (abs(mx - mxi) > 1) or (abs(my - myi) > 1) or \
               (abs(mz - mzi) > 1):
                msg = "\tSpline warning: spline location outside grid cell"
                msg += "with maximum coalescence value"
                self.output.log(msg, self.log)

            xyz = self.lut.xyz2loc(np.array([[mxi, myi, mzi]]), inverse=True)
            loc_spline = self.lut.xyz2coord(xyz)[0]

            # Run check that spline location is within window
            if (abs(mx - mxi) > w2) or (abs(my - myi) > w2) or \
               (abs(mz - mzi) > w2):
                msg = "\t !!!! Spline error: location outside interpolation "
                msg += "window !!!!\n\t\t\tGridded Location returned"
                self.output.log(msg, self.log)

                xyz = self.lut.xyz2loc(np.array([[mx, my, mz]]), inverse=True)
                loc_spline = self.lut.xyz2coord(xyz)[0]

        else:
            msg = "\t !!!! Spline error: interpolation window crosses edge of "
            msg += "grid !!!!\n\t\t\tGridded Location returned"
            self.output.log(msg, self.log)

            xyz = self.lut.xyz2loc(np.array([[mx, my, mz]]), inverse=True)
            loc_spline = self.lut.xyz2coord(xyz)[0]

        return loc_spline

    def _calculate_location(self, map_4d):
        """
        Marginalise 4-D coalescence grid. Calcuate a set of locations and
        associated uncertainties by:
            (1) calculating the covariance of the entire coalescence map;
            (2) fitting a 3-D Gaussian function and ..
            (3) a 3-D spline function ..
                to a region around the maximum coalescence location in the
                marginalised 3-D coalescence map.

        Parameters
        ----------
        map_4d : array-like
            4-D coalescence grid output from _compute()

        Returns
        -------
        loc_spline: array-like
            [x, y, z] : expectation location from local spline interpolation

        loc_gau : array-like
            [x, y, z] : best-fit location from local gaussian fit to the
                        coalescence grid

        loc_gau_err : array-like
            [x_err, y_err, z_err] : one sigma uncertainties associated with
                                    loc_gau

        loc_cov : array-like
            [x, y, z] : best-fit location from covariance fit over entire 3d
                        grid (by default after filtering above a certain
                        percentile).

        loc_cov_err : array-like
            [x_err, y_err, z_err] : one sigma uncertainties associated with
                                    loc_cov

        """

        # MARGINALISE: Determining the 3-D coalescence map
        self.coa_map = np.log(np.sum(np.exp(map_4d), axis=-1))

        # Normalise
        self.coa_map = self.coa_map/np.max(self.coa_map)

        # Fit 3-D spline function to small window around max coalescence
        # location and interpolate to determine sub-grid maximum coalescence
        # location.
        loc_spline = self._splineloc(np.copy(self.coa_map))

        # Apply gaussian smoothing to small window around max coalescence
        # location and fit 3-D gaussian function to determine local
        # expectation location and uncertainty
        smoothed_coa_map = self._gaufilt3d(np.copy(self.coa_map))
        loc_gau, loc_gau_err = self._gaufit3d(np.copy(smoothed_coa_map),
                                              thresh=0.)

        # Calculate global covariance expected location and uncertainty
        loc_cov, loc_cov_err = self._covfit3d(np.copy(self.coa_map))

        return loc_spline, loc_gau, loc_gau_err, loc_cov, loc_cov_err

    def _empty(self, w_beg, w_end):
        """
        Create an empty set of arrays to write to .scanmseed ; used where there
        is no data available to run _compute() .

        Parameters
        ----------
        w_beg : UTCDateTime object
            Start time to create empty arrays

        w_end : UTCDateTime object
            End time to create empty arrays

        Returns
        -------
        daten, max_coa, max_coa_norm, coord : array-like
            Empty arrays with the correct shape to write to .scanmseed as if
            they were coastream outputs from _compute()

        """

        tmp = np.arange(w_beg + self.pre_pad,
                        w_end - self.post_pad + (1 / self.sampling_rate),
                        1 / self.sampling_rate)
        daten = [x.datetime for x in tmp]

        max_coa = max_coa_norm = np.full(len(daten), 0)

        coord = np.full((len(daten), 3), 0)

        return daten, max_coa, max_coa_norm, coord

    def _optional_locate_outputs(self, event_mw_data, event, out_str,
                                 phase_picks, event_uid, map_4d):
        """
        Deal with optional outputs in locate():
            plot_event_summary()
            plot_station_traces()
            plot_coal_video()
            write_cut_waveforms()
            write_4d_coal_grid()

        Parameters
        ----------
        event_mw_data : pandas DataFrame
            Gridded maximum coa location through time across the marginal
            window. Columns = ["DT", "COA", "X", "Y", "Z"]

        event : pandas DataFrame
            Final event location information.
            Columns = ["DT", "COA", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ"]
            All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres

        out_str : str
            String {run_name}_{event_name} (figure displayed by default)

        phase_picks: dict
            Phase pick info, with keys:
                "Pick" : pandas DataFrame
                "GAU_P" : array-like, dict
                "GAU_S" : array-like, dict

        event_uid : str
            UID of earthquake: "YYYYMMDDHHMMSSFFFF"

        map_4d : array-like
            4-D coalescence grid output from _compute()

        """

        if self.plot_event_summary or self.plot_station_traces or \
           self.plot_coal_video:
            quake_plot = qplot.QuakePlot(self.lut, self.data, event_mw_data,
                                         self.marginal_window, self.output.run,
                                         event, phase_picks, map_4d,
                                         self.coa_map)

        if self.plot_event_summary:
            timer = util.Stopwatch()
            self.output.log("\tPlotting event summary figure...", self.log)
            quake_plot.event_summary(file_str=out_str)
            self.output.log(timer(), self.log)

        if self.plot_station_traces:
            timer = util.Stopwatch()
            self.output.log("\tPlotting station traces...", self.log)
            quake_plot.station_traces(file_str=out_str, event_name=event_uid)
            self.output.log(timer(), self.log)

        if self.plot_coal_video:
            timer = util.Stopwatch()
            self.output.log("\tPlotting coalescence video...", self.log)
            quake_plot.coalescence_video(file_str=out_str)
            self.output.log(timer(), self.log)

        if self.write_cut_waveforms:
            self.output.log("\tSaving raw cut waveforms...", self.log)
            timer = util.Stopwatch()
            self.output.write_cut_waveforms(self.data, event, event_uid,
                                            self.cut_waveform_format,
                                            self.pre_cut, self.post_cut)
            self.output.log(timer(), self.log)

        if self.write_4d_coal_grid:
            timer = util.Stopwatch()
            self.output.log("\tSaving 4D coalescence grid...", self.log)
            t_beg = UTCDateTime(event_mw_data["DT"].iloc[0])
            t_end = UTCDateTime(event_mw_data["DT"].iloc[-1])
            self.output.write_coal4D(map_4d, event_uid, t_beg, t_end)
            self.output.log(timer(), self.log)

        del quake_plot
