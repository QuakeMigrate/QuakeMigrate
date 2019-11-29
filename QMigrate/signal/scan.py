# -*- coding: utf-8 -*-
"""
Module to perform core QuakeMigrate functions: detect() and locate().

"""

import warnings

import numpy as np
from obspy import UTCDateTime, Stream, Trace
from obspy.geodetics.base import gps2dist_azimuth
from obspy.io.xseed import Parser
from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp, evalresp_for_frequencies
import pandas as pd
from scipy.interpolate import Rbf
from scipy.signal import fftconvolve, find_peaks

import QMigrate.core.QMigratelib as ilib
import QMigrate.io.quakeio as qio
import QMigrate.plot.quakeplot as qplot
import QMigrate.signal.magnitudes as qmag
import QMigrate.signal.onset.onset as qonset
import QMigrate.signal.pick.pick as qpick
import QMigrate.util as util

# Filter warnings
warnings.filterwarnings("ignore", message=("Covariance of the parameters"
                                           " could not be estimated"))
warnings.filterwarnings("ignore", message=("File will be written with more"
                                           " than one different record"
                                           " lengths. This might have a"
                                           " negative influence on the"
                                           " compatibility with other"
                                           " programs."))


class DefaultQuakeScan(object):
    """
    Default parameter class for QuakeScan.

    """

    def __init__(self):
        """
        Initialise DefaultQuakeScan object.

        Parameters (all optional)
        ----------
        time_step : float
            Time length (in seconds) of time step used in detect(). Note: total
            detect run duration should be divisible by time_step. Increasing
            time_step will increase RAM usage during detect, but will slightly
            speed up overall detect run.

        sampling_rate : int
            Desired sampling rate for input data; sampling rate that detect()
            and locate() will be computed at.

        marginal_window : float
            Time window (+/- marginal_window) about the maximum coalescence
            time to marginalise the 4-D coalescence grid computed in locate()
            to estimate the earthquake location and uncertainty. Should be an
            estimate of the time uncertainty in the earthquake origin time -
            a combination of the expected spatial error and the seismic
            velocity in the region of the event

        n_cores : int
            Number of cores to use on the executing host for detect() /locate()

        continuous_scanmseed_write : bool
            Option to continuously write the .scanmseed file outputted by
            detect() at the end of every time step. Default behaviour is to
            write at the end of the time period, or the end of each day.

        plot_event_summary : bool, optional
            Plot event summary figure - see QMigrate.plot.quakeplot for more
            details.

        plot_event_video : bool, optional
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

        # Time step for continuous compute in detect
        self.time_step = 120.

        # Data sampling rate
        self.sampling_rate = 50

        # Marginal window
        self.marginal_window = 2.

        # Number of cores to perform detect/locate on
        self.n_cores = 1

        # Toggle whether to incrementally write .scanmseed in detect()
        self.continuous_scanmseed_write = False

        # Plotting toggles
        self.plot_event_summary = True
        self.plot_event_video = False

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

    EVENT_FILE_COLS = ["DT", "COA", "COA_NORM", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ", "TRIG_COA", "DEC_COA",
                       "DEC_COA_NORM", "ML", "ML_Err"]

    def __init__(self, data, lut, onset, picker=None, get_amplitudes=False,
                 calc_mag=False, quick_amplitudes=False, get_polarities=False,
                 output_path=None, run_name=None, log=False):
        """
        Class initialisation method.

        Parameters
        ----------
        data : Archive object
            Contains information on data archive structure and
            read_waveform_data() method

        lut : LUT object
            Contains the travel-time lookup tables for P- and S-phases,
            computed for some pre-defined velocity model

        onset : Onset object
            Contains definitions for the P- and S-phase onset functions

        picker : PhasePicker object
            Wraps methods to perform phase picking on the seismic data

        output_path : str
            Path of parent output directory: e.g. ./OUTPUT

        run_name : str
            Name of current run: all outputs will be saved in the directory
            output_path/run_name

        log : bool, optional
            Toggle for logging - default is to print all information to stdout.
            If True, will also create a log file.

        """

        DefaultQuakeScan.__init__(self)

        self.data = data
        self.lut = lut

        if output_path is not None:
            self.output = qio.QuakeIO(output_path, run_name, log)
        else:
            self.output = None

        if isinstance(onset, qonset.Onset):
            self.onset = onset
        else:
            raise util.OnsetTypeError

        if picker is None:
            pass
        elif isinstance(picker, qpick.PhasePicker):
            self.picker = picker
            self.picker.lut = lut
            self.picker.output = self.output
        else:
            raise util.PickerTypeError

        self.polarity = get_polarities
        self.quick_amps = quick_amplitudes
        if calc_mag:
            self.amplitudes = True
            self.amplitude_params = {}
            self.calc_mag = True
            self.magnitude_params = {}
        elif get_amplitudes:
            self.amplitudes = True
            self.amplitude_params = {}
            self.calc_mag = False
            self.magnitude_params = None
        else:
            self.amplitudes = False
            self.amplitude_params = None
            self.calc_mag = False
            self.magnitude_params = None

        # Create Wood Anderson response - different to the ObsPy values
        # http://www.iris.washington.edu/pipermail/sac-help/2013-March/001430.html
        self.WOODANDERSON = {"poles": [-5.49779 + 5.60886j,
                                       -5.49779 - 5.60886j],
                             "zeros": [0j, 0j],
                             "sensitivity": 2080,
                             "gain": 1.}

        self.log = log

        # Get pre- and post-pad values from the onset class
        self.pre_pad = self.onset.pre_pad
        self.onset.post_pad = lut.max_ttime
        self.post_pad = self.onset.post_pad

        msg = ("=" * 110 + "\n") * 2
        msg += "\tQuakeMigrate - Coalescence Scanning - Path: {} - Name: {}\n"
        msg += ("=" * 110 + "\n") * 2
        msg = msg.format(self.output.path, self.output.name)
        self.output.log(msg, self.log)

    def __str__(self):
        """
        Return short summary string of the QuakeScan object

        It will provide information on all of the various parameters that the
        user can/has set.

        """

        out = "Scan parameters:"
        out += "\n\tTime step\t\t:\t{}".format(self.time_step)
        out += "\n\n\tData sampling rate\t:\t{}".format(self.sampling_rate)
        out += "\n\n\tPre-pad\t\t\t:\t{}".format(self.pre_pad)
        out += "\n\tPost-pad\t\t:\t{}".format(self.post_pad)
        out += "\n\n\tMarginal window\t\t:\t{}".format(self.marginal_window)
        out += "\n\n\tNumber of CPUs\t\t:\t{}".format(self.n_cores)

        return out

    def detect(self, start_time, end_time):
        """
        Scans through continuous data calculating coalescence on a (decimated)
        3D grid by back-migrating P and S onset (characteristic) functions.

        Parameters
        ----------
        start_time : str
            Start time of continuous scan

        end_time : str
            End time of continuous scan (last sample returned will be that
            which immediately precedes this time stamp)

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        msg = "=" * 110 + "\n"
        msg += "\tDETECT - Continuous Seismic Processing\n"
        msg += "=" * 110 + "\n\n"
        msg += "\tParameters:\n"
        msg += "\t\tStart time     = {}\n"
        msg += "\t\tEnd   time     = {}\n"
        msg += "\t\tTime step (s)  = {}\n"
        msg += "\t\tNumber of CPUs = {}\n\n"
        msg += str(self.onset)
        msg += "=" * 110
        msg = msg.format(str(start_time), str(end_time), self.time_step,
                         self.n_cores, self.sampling_rate)
        self.output.log(msg, self.log)

        # Detect max coalescence value and location at each time sample
        # within the (decimated) grid
        self._continuous_compute(start_time, end_time)

    def locate(self, fname=None, start_time=None, end_time=None):
        """
        Re-computes the 3D coalescence on an undecimated grid for a short
        time window around each candidate earthquake triggered from the
        (decimated) continuous detect scan. Calculates event location and
        uncertainties, makes phase arrival picks, plus multiple optional
        plotting / data outputs for further analysis and processing.

        Parameters
        ----------
        fname : str
            Filename of triggered events that will be located

        start_time : str
            Start time of locate run: earliest event trigger time that will be
            located

        end_time : str
            End time of locate run: latest event trigger time that will be
            located is one sample before this time

        """

        if not fname and not start_time and not end_time:
            raise RuntimeError("Must supply an input argument.")

        if not fname:
            # Convert times to UTCDateTime objects
            start_time = UTCDateTime(start_time)
            end_time = UTCDateTime(end_time)

        msg = "=" * 110 + "\n"
        msg += "\tLOCATE - Determining earthquake location and uncertainty\n"
        msg += "=" * 110 + "\n\n"
        msg += "\tParameters:\n"
        msg += "\t\tStart time     = {}\n"
        msg += "\t\tEnd   time     = {}\n"
        msg += "\t\tNumber of CPUs = {}\n\n"
        msg += str(self.onset)
        msg += str(self.picker)
        msg += "=" * 110 + "\n"
        msg = msg.format(str(start_time), str(end_time), self.n_cores)
        self.output.log(msg, self.log)

        if fname:
            self._locate_events(fname)
        else:
            self._locate_events(start_time, end_time)

    def _append_coastream(self, coastream, daten, max_coa, max_coa_norm, loc):
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
                "sampling_rate": self.sampling_rate,
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

            coastream.trim(starttime=write_end + 1 / self.sampling_rate)

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

        # Initialise coastream object
        coastream = None

        t_length = self.pre_pad + self.post_pad + self.time_step
        self.pre_pad += np.ceil(t_length * 0.06)
        self.pre_pad = int(np.ceil(self.pre_pad * self.sampling_rate)
                           / self.sampling_rate * 1000) / 1000
        self.post_pad += np.ceil(t_length * 0.06)
        self.post_pad = int(np.ceil(self.post_pad * self.sampling_rate)
                            / self.sampling_rate * 1000) / 1000

        try:
            nsteps = int(np.ceil((end_time - start_time) / self.time_step))
        except AttributeError:
            msg = "Error: Time step has not been specified"
            self.output.log(msg, self.log)
            return

        # Initialise pandas DataFrame object to track availability
        stn_ava_data = pd.DataFrame(index=np.arange(nsteps),
                                    columns=self.data.stations)

        for i in range(nsteps):
            timer = util.Stopwatch()
            w_beg = start_time + self.time_step * i - self.pre_pad
            w_end = start_time + self.time_step * (i + 1) + self.post_pad

            msg = ("~" * 19) + " Processing : {} - {} " + ("~" * 19)
            msg = msg.format(str(w_beg), str(w_end))
            self.output.log(msg, self.log)

            try:
                self.data.read_waveform_data(w_beg, w_end, self.sampling_rate)
                daten, max_coa, max_coa_norm, coord, _ = self._compute(
                                                              w_beg, w_end)
                stn_ava_data.loc[i] = self.data.availability

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
                                                        coord[:-1, :])

            del daten, max_coa, max_coa_norm, coord

            if self.continuous_scanmseed_write and not written:
                self.output.write_coastream(coastream)
                written = True

            self.output.log(timer(), self.log)

        if not written:
            self.output.write_coastream(coastream)

        del coastream

        self.output.write_stn_availability(stn_ava_data)
        self.output.log("=" * 110, self.log)

    def _locate_events(self, *args):
        """
        Loop through list of earthquakes read in from trigger results and
        re-compute coalescence; output phase picks, event location and
        uncertainty, plus optional plots and outputs.

        Parameters
        ----------

        args :
            A variable length tuple holding either a start_time - end_time pair
            or a filename a triggered events to read

        start_time : UTCDateTime object
           Start time of locate run: earliest event trigger time that will be
            located

        end_time : UTCDateTime object
            End time of locate run: latest event trigger time that will be
            located

        fname : str
            File of triggered events to read

        """

        if len(args) == 2:
            start_time = args[0]
            end_time = args[1]
            trig_events = self.output.read_triggered_events_time(start_time,
                                                                 end_time).reset_index()
        elif len(args) == 1:
            fname = args[0]
            trig_events = self.output.read_triggered_events(fname).reset_index()
        n_evts = len(trig_events)

        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.p_onset_win[1],
                               self.s_onset_win[1]) \
                           + 3 * max(self.p_onset_win[0],
                                     self.s_onset_win[0])

        # Adjust pre- and post-pad to take into account cosine taper
        t_length = self.pre_pad + 4*self.marginal_window + self.post_pad
        self.pre_pad += np.ceil(t_length * 0.06)
        self.pre_pad = int(np.ceil(self.pre_pad * self.sampling_rate)
                           / self.sampling_rate * 1000) / 1000
        self.post_pad += np.ceil(t_length * 0.06)
        self.post_pad = int(np.ceil(self.post_pad * self.sampling_rate)
                            / self.sampling_rate * 1000) / 1000

        for i, trig_event in trig_events.iterrows():
            event_uid = trig_event["EventID"]
            msg = "=" * 110 + "\n"
            msg += "\tEVENT - {} of {} - {}\n"
            msg += "=" * 110 + "\n\n"
            msg += "\tDetermining event location...\n"
            msg = msg.format(i + 1, n_evts, event_uid)
            self.output.log(msg, self.log)

            trig_coa = trig_event["COA_V"]
            try:
                detect_coa = trig_event["COA"]
                detect_coa_norm = trig_event["COA_NORM"]
            except KeyError:
                detect_coa = np.nan
                detect_coa_norm = np.nan

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

            daten, max_coa, max_coa_norm, coord, map_4d = self._compute(
                                                               w_beg, w_end)
            event_coa_data = pd.DataFrame(np.array((daten, max_coa,
                                                    max_coa_norm,
                                                    coord[:, 0],
                                                    coord[:, 1],
                                                    coord[:, 2])).transpose(),
                                          columns=["DT", "COA", "COA_NORM",
                                                   "X", "Y", "Z"])
            event_coa_data["DT"] = event_coa_data["DT"].apply(UTCDateTime)
            idxmax = event_coa_data["COA"].astype("float").idxmax()
            event_coa_data_dtmax = event_coa_data["DT"].iloc[idxmax]
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
                msg += "\n" + "=" * 110 + "\n"
                msg = msg.format(event_uid)
                self.output.log(msg, self.log)
                continue

            event_mw_data = event_coa_data
            event_mw_data = event_mw_data[(event_mw_data["DT"] >= w_beg_mw) &
                                          (event_mw_data["DT"] <= w_end_mw)]
            map_4d = map_4d[:, :, :,
                            event_mw_data.index[0]:event_mw_data.index[-1]]
            event_mw_data = event_mw_data.reset_index(drop=True)
            idxmax = event_mw_data["COA"].astype("float").idxmax()
            event_max_coa = event_mw_data.iloc[idxmax]

            out_str = "{}_{}".format(self.output.name, event_uid)
            self.output.log(timer(), self.log)

            # Make phase picks
            timer = util.Stopwatch()
            self.output.log("\tMaking phase picks...", self.log)
            self.picker.pick_phases(self.data, event_max_coa)
            self.picker.write_picks(event_uid)
            self.output.log(timer(), self.log)

            # Determining earthquake location error
            timer = util.Stopwatch()
            self.output.log("\tDetermining earthquake location and uncertainty...",
                            self.log)
            loc_spline, loc_gau, loc_gau_err, loc_cov, \
                loc_cov_err = self._calculate_location(map_4d)
            self.output.log(timer(), self.log)

            # Determine amplitudes
            if self.amplitudes:
                self.output.log("\tGetting amplitudes...", self.log)
                amps = self._get_amplitudes(event_max_coa.values[0],
                                            loc_gau)
                self.output.write_amplitudes(amps, event_uid)

            if self.amplitudes and self.calc_mag:
                self.output.log("\tCalculating magnitude...", self.log)
                mags = qmag.calc_magnitude(amps, self.magnitude_params)
                self.output.write_amplitudes(mags, event_uid)
                ML, ML_error = qmag.mean_magnitude(mags, self.magnitude_params)
                self.output.log(timer(), self.log)
            else:
                ML = np.nan
                ML_error = np.nan

            # Make event dictionary with all final event location data
            event = pd.DataFrame([[event_max_coa.values[0],
                                   event_max_coa.values[1],
                                   event_max_coa.values[2],
                                   loc_spline[0], loc_spline[1], loc_spline[2],
                                   loc_gau[0], loc_gau[1], loc_gau[2],
                                   loc_gau_err[0], loc_gau_err[1],
                                   loc_gau_err[2],
                                   loc_cov[0], loc_cov[1], loc_cov[2],
                                   loc_cov_err[0], loc_cov_err[1],
                                   loc_cov_err[2], trig_coa, detect_coa,
                                   detect_coa_norm, ML, ML_error]],
                                 columns=self.EVENT_FILE_COLS)

            self.output.write_event(event, event_uid)

            self._optional_locate_outputs(event_mw_data, event, out_str,
                                          event_uid, map_4d)

            self.output.log("=" * 110 + "\n", self.log)

            del map_4d, event_coa_data, event_mw_data, event_max_coa
            self.coa_map = None

    def _get_amplitudes(self, otime, coord):
        """
        Measure the amplitudes for the purpose of magnitude calculation.

        Parameters
        ----------
        otime : obspy UTCDateTime object
            Origin time of earthquake.

        coord : array-like
            Coordinate of earthquake in the input projection space.

        Returns
        -------
        amps : pandas DataFrame object


        """

        if not self.amplitude_params:
            msg = ("Must define a set of amplitude parameters.\n"
                   "For more information, please consult the documentation.")
            raise AttributeError(msg)

        if self.amplitude_params["response_format"] == "dataless":
            dataless = Parser(self.amplitude_params["response_fname"])
            resp = False
        else:
            resp_file = self.amplitude_params["response_fname"]
            resp = True

        if not self.quick_amps:
            water_level = self.amplitude_params.get("water_level")
            pre_filt = self.amplitude_params.get("pre_filt")
        # else:
        #     inv = read_inventory(self.amplitude_params["dataless_xml"])

        noise_window = self.amplitude_params.get("noise_win")

        evlo, evla, evdp = coord
        picks = self.picker.phase_picks["Pick"]
        stations = self.lut.station_data

        raw_waveforms = self.data.raw_waveforms.copy()

        amplitudes = pd.DataFrame(columns=["id", "epi_distance", "depth",
                                           "P_amp", "P_freq", "S_amp",
                                           "S_freq", "Error", "is_picked"])

        for i, pick in picks.iterrows():
            p = pick["PickTime"]
            picked = True
            if p == -1:
                p = pick["ModelledTime"]
                picked = not picked

            # Take advantage of Python's function scope - state of variables pp
            # and ppicked are retained when going to next loop
            if pick["Phase"] == "P":
                pp = UTCDateTime(p)
                ppicked = picked
                continue
            elif pick["Phase"] == "S":
                sp = UTCDateTime(p)
                spicked = picked
            picked = (ppicked or spicked)

            station = pick["Name"]
            station_info = stations[stations["Name"] == station]
            stla = station_info["Latitude"].iloc[0]
            stlo = station_info["Longitude"].iloc[0]
            stel = station_info["Elevation"].iloc[0]

            # Evaluate epicentral/vertical distances between station/event
            edist, *_ = gps2dist_azimuth(evla, evlo, stla, stlo) / 1000.
            zdist = (evdp - stel) / 1000.

            amps_template = ["", edist, zdist, np.nan, np.nan, np.nan, np.nan,
                             np.nan, False]

            st = raw_waveforms.select(station=station)
            st.merge(method=1, fill_value="interpolate")
            if not bool(st) or len(st) > 3:
                print("No data available for {}.".format(station))
                amps = amps_template.copy()
                for j, channel in enumerate(["N", "E", "Z"]):
                    amps[0] = ".{}..{}".format(station, channel)
                    amplitudes.loc[i//2+j] = amps
                    continue

            if not self.quick_amps:
                if not resp:
                    st.simulate(seedresp={"filename": dataless,
                                          "units": "DIS"},
                                paz_simulate=self.WOODANDERSON,
                                pre_filt=pre_filt, taper=False,
                                water_level=water_level)
                else:
                    tmp = Stream()
                    for tr in st:
                        tr.simulate(seedresp={"filename": resp_file,
                                              "units": "DIS"},
                                    paz_simulate=self.WOODANDERSON,
                                    pre_filt=pre_filt, taper=False,
                                    water_level=water_level)
                        tmp += tr
                    st = tmp

            for j, channel in enumerate(["N", "E", "Z"]):
                amps = amps_template.copy()
                tr = st.select(channel="*{}".format(channel))[0]
                stats = tr.stats
                if bool(tr):
                    print("No data for {} component.".format(channel))
                    amps[0] = ".{}..{}".format(station, channel)
                    amplitudes.loc[i//2+j] = amps
                    continue

                amps[0] = tr.id

                assert pp < sp
                if np.abs(pp - sp) < 1.:
                    windows = [[pp, sp], [sp, sp + noise_window]]
                else:
                    windows = [[pp, sp - 1.], [sp - 1., sp + noise_window]]

                for k, (stime, etime) in enumerate(windows):
                    # Add 5% for tapering
                    taper = (etime - stime) * 0.05
                    window = tr.slice(stime - taper, etime + taper)
                    data = window.data

                    if bool(window) or np.all(data == 0.0) or np.all(data == data.mean()):
                        continue

                    half_amp, approx_freq = self._peak_to_trough_amplitude(window)

                    if self.quick_amps:
                        gain = paz_2_amplitude_value_of_freq_resp(self.WOODANDERSON,
                                                                  approx_freq) * self.WOODANDERSON["sensitivity"]

                        if not resp:
                            # Get the response from the dataless SEED volume
                            blockettes = dataless._select(tr.id,
                                                          datetime=stats.starttime)
                            response = dataless.get_response_for_channel(blockettes[1:], "")
                            gain /= np.abs(response.get_evalresp_response_for_frequencies([approx_freq],
                                                                                          output="DISP"))
                        else:
                            try:
                                gain /= np.abs(evalresp_for_frequencies(stats.delta,
                                                                        [approx_freq],
                                                                        resp_file,
                                                                        stats.starttime,
                                                                        units="DISP",
                                                                        station=stats.station,
                                                                        channel=stats.channel))
                            except ValueError:
                                continue
                        half_amp *= gain

                    amps[3+k*2:5+k*2] = half_amp * 1000., approx_freq

                # Grab a noise window
                data = tr.slice(pp - 3. - noise_window, pp - 3.)
                noise = np.std(data.data)
                if self.quick_amps:
                    noise *= gain
                amps[7:9] = noise * 2., picked

                amplitudes[i//2+j] = amps

        return amplitudes.set_index("id")

    def _peak_to_trough_amplitude(self, trace):
        """
        Calculate the peak-to-trough amplitude for a given trace.

        Parameters
        ----------
        trace : ObsPy Trace object
            Waveform for which to calculate peak-to-trough amplitude.

        Returns
        -------
        half_amp : float
            Half the value of maximum peak-to-trough amplitude.
            Returns -1 if no measurement could be made.

        approx_freq : float
            Approximate frequency of the arrival, based on the half-period
            between the maximum peak/trough.
            Returns -1 if no measurement could be made.

        """

        try:
            prom_mult = self.amplitude_params["prominence_multiplier"]
        except KeyError:
            prom_mult = 0.

        trace.detrend("linear")
        trace.taper(0.05)

        prominence = prom_mult * np.max(np.abs(trace.data))
        peaks, _ = find_peaks(trace.data, prominence=prominence)
        troughs, _ = find_peaks(-trace.data, prominence=prominence)

        if len(peaks) == 0 or len(troughs) == 0:
            return -1, -1
        elif len(peaks) == 1 and len(troughs) == 1:
            full_amp = np.abs(trace.data[peaks] - trace.data[troughs])
            pos = 0
        elif len(peaks) == len(troughs):
            if peaks[0] < troughs[0]:
                a, b, c, d = peaks, troughs, peaks[1:], troughs[:-1]
            else:
                a, b, c, d = peaks, troughs, peaks[:-1], troughs[1:]
        elif not np.abs(len(peaks) - len(troughs)) == 1:
            # More than two peaks/troughs next to one another
            return -1, -1
        elif len(peaks) > len(troughs):
            assert peaks[0] < troughs[0]
            a, b, c, d = peaks[:-1], troughs, peaks[1:], troughs
        elif len(peaks) < len(troughs):
            assert peaks[0] > troughs[0]
            a, b, c, d = peaks, troughs[1:], peaks, troughs[:-1]

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

        peak_t = trace.times()[peaks[pos]]
        trough_t = trace.times()[troughs[pos]]
        approx_freq = 1. / (np.abs(peak_t - trough_t) * 2.)

        return full_amp / 2, approx_freq

    def _read_event_waveform_data(self, event, w_beg, w_end):
        """
        Read waveform data for a triggered event.

        Parameters
        ----------
        event : pandas DataFrame
            Triggered event output from _trigger_scn().
            Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                      "COA_Z", "MinTime", "MaxTime"]

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

    def _compute(self, w_beg, w_end):
        """
        Compute 3-D coalescence between two time stamps.

        Parameters
        ----------
        w_beg : UTCDateTime object
            Time stamp of first sample in window

        w_end : UTCDateTime object
            Time stamp of final sample in window

        Returns
        -------
        daten : array-like
            UTCDateTime time stamp for each sample between w_beg and w_end

        max_coa : array-like
            Coalescence value through time

        max_coa_norm : array-like
            Normalised coalescence value through time

        coord : array-like
            Location of maximum coalescence through time in input projection
            space.

        map_4d : array-like
            4-D coalescence map

        """

        avail_idx = np.where(self.data.availability == 1)[0]

        onsets = self.onset.calculate_onsets(self.data)
        nchan, tsamp = onsets.shape

        ttimes = self.lut.ttimes(self.sampling_rate)

        # Calculate no. of samples in the pre-pad, post-pad and main window
        pre_smp = int(round(self.pre_pad * int(self.sampling_rate)))
        pos_smp = int(round(self.post_pad * int(self.sampling_rate)))
        nsamp = tsamp - pre_smp - pos_smp

        # Prep empty 4-D coalescence map and run C-compiled ilib.migrate()
        ncell = tuple(self.lut.cell_count)
        map_4d = np.zeros(ncell + (nsamp,), dtype=np.float64)
        ilib.migrate(onsets, ttimes, pre_smp, pos_smp, nsamp, map_4d,
                     self.n_cores)

        # Prep empty coalescence and unraveled grid index arrays and run
        # C-compiled ilib.find_max_coa()
        max_coa = np.zeros(nsamp, np.double)
        max_coa_idx = np.zeros(nsamp, np.int64)
        ilib.find_max_coa(map_4d, max_coa, max_coa_idx, 0, nsamp, self.n_cores)

        # Get max_coa_norm
        max_coa_norm = max_coa / np.sum(map_4d, axis=(0, 1, 2))
        max_coa_norm = max_coa_norm * map_4d.shape[0] * map_4d.shape[1] * \
            map_4d.shape[2]

        tmp = np.arange(w_beg + self.pre_pad,
                        w_end - self.post_pad + (1 / self.sampling_rate),
                        1 / self.sampling_rate)
        daten = [x.datetime for x in tmp]

        # Calculate max_coa (with correction for number of stations)
        max_coa = np.exp((max_coa / (len(avail_idx) * 2)) - 1.0)

        # Convert the flat grid indices (of maximum coalescence) to coordinates
        # in the input projection space.
        coord = self.lut.index2coord(max_coa_idx, unravel=True)

        return daten, max_coa, max_coa_norm, coord, map_4d

    def _gaufilt3d(self, coa_map, sgm=0.8, shp=None):
        """
        Smooth the 3-D marginalised coalescence map using a 3-D Gaussian
        function to enable a better Gaussian fit to the data to be calculated.

        Parameters
        ----------
        coa_map : array-like
            Marginalised 3-D coalescence map

        sgm : float
            Sigma value (in grid cells) for the 3-D Gaussian filter function;
            bigger sigma leads to more aggressive (long wavelength) smoothing

        shp : array-like, optional
            Shape of volume

        Returns
        -------
        smoothed_map_3d : array-like
            Gaussian smoothed 3-D coalescence map

        """

        if shp is None:
            shp = coa_map.shape
        nx, ny, nz = shp

        # Construct 3-D Gaussian filter
        flt = util.gaussian_3d(nx, ny, nz, sgm)

        # Convolve map_3d and 3-D Gaussian filter
        smoothed_coa_map = fftconvolve(coa_map, flt, mode="same")

        # Mirror and convolve again (to avoid "phase-shift")
        smoothed_coa_map = smoothed_coa_map[::-1, ::-1, ::-1] \
            / np.nanmax(smoothed_coa_map)
        smoothed_coa_map = fftconvolve(smoothed_coa_map, flt, mode="same")

        # Final mirror and normalise
        smoothed_coa_map = smoothed_coa_map[::-1, ::-1, ::-1] \
            / np.nanmax(smoothed_coa_map)

        return smoothed_coa_map

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
            Marginalised 3-D coalescence map.

        thresh : float (between 0 and 1), optional
            Cut-off threshold (fractional percentile) to trim coa_map; only
            data above this percentile will be retained.

        win : int, optional
            Window of grid cells (+/-(win-1)//2 in x, y and z) around max
            value in coa_map to perform the fit over.

        Returns
        -------
        location : array-like, [x, y, z]
            Expectation location from covariance fit.

        uncertainty : array-like, [sx, sy, sz]
            One sigma uncertainties on expectation location from covariance
            fit.

        """

        # Get shape of 3-D coalescence map and max coalesence grid location
        shape = coa_map.shape
        ijk = np.unravel_index(np.nanargmax(coa_map), coa_map.shape)

        # If window is specified, clip the grid to only look here.
        if win:
            flag = np.logical_and(coa_map > thresh, self._mask3d(shape, ijk,
                                                                 win))
        else:
            flag = np.where(coa_map > thresh, True, False)

        # Treat the coalescence values in the grid as the sample weights
        sw = coa_map.flatten()
        sw[~flag.flatten()] = np.nan
        ssw = np.nansum(sw)

        # Get the x, y and z samples on which to perform the fit
        cc = self.lut.cell_count
        cs = self.lut.cell_size
        grid = np.meshgrid(np.arange(cc[0]), np.arange(cc[1]),
                           np.arange(cc[2]), indexing="ij")
        xs, ys, zs = [g.flatten() * size for g, size in zip(grid, cs)]

        # Expectation values:
        xe, ye, ze = [np.nansum(sw * s) / ssw for s in [xs, ys, zs]]

        # Covariance matrix:
        cov_matrix = np.zeros((3, 3))
        cov_matrix[0, 0] = np.nansum(sw * (xs - xe) ** 2) / ssw
        cov_matrix[1, 1] = np.nansum(sw * (ys - ye) ** 2) / ssw
        cov_matrix[2, 2] = np.nansum(sw * (zs - ze) ** 2) / ssw
        cov_matrix[0, 1] = np.nansum(sw * (xs - xe) * (ys - ye)) / ssw
        cov_matrix[1, 0] = cov_matrix[0, 1]
        cov_matrix[0, 2] = np.nansum(sw * (xs - xe) * (zs - ze)) / ssw
        cov_matrix[2, 0] = cov_matrix[0, 2]
        cov_matrix[1, 2] = np.nansum(sw * (ys - ye) * (zs - ze)) / ssw
        cov_matrix[2, 1] = cov_matrix[1, 2]

        location_xyz = self.lut.ll_corner + np.array([xe, ye, ze])
        location = self.lut.coord2grid(location_xyz, inverse=True)[0]
        uncertainty = np.diag(np.sqrt(abs(cov_matrix)))

        return location, uncertainty

    def _gaufit3d(self, coa_map, thresh=0., win=7):
        """
        Fit a 3-D Gaussian function to a region around the maximum coalescence
        location in the 3-D marginalised coalescence map: return expectation
        location and associated uncertainty.

        Parameters
        ----------
        coa_map : array-like
            Marginalised 3-D coalescence map.

        thresh : float (between 0 and 1), optional
            Cut-off threshold (percentile) to trim coa_map: only data above
            this percentile will be retained.

        win : int, optional
            Window of grid cells (+/-(win-1)//2 in x, y and z) around max
            value in coa_map to perform the fit over.

        Returns
        -------
        location : array-like, [x, y, z]
            Expectation location from 3-D Gaussian fit.

        uncertainty : array-like, [sx, sy, sz]
            One sigma uncertainties on expectation location from 3-D Gaussian
            fit.

        """

        # Get shape of 3-D coalescence map and max coalesence grid location
        shape = coa_map.shape
        ijk = np.unravel_index(np.nanargmax(coa_map), shape)

        # Only use grid cells above threshold value, and within the specified
        # window around the coalescence peak
        flag = np.logical_and(coa_map > thresh, self._mask3d(shape, ijk, win))
        ix, iy, iz = np.where(flag)

        # Subtract mean of entire 3-D coalescence map from the local grid
        # window so it is better approximated by a Gaussian (which goes to zero
        # at infinity)
        coa_map = coa_map - np.nanmean(coa_map)

        # Fit 3-D Gaussian function
        ncell = len(ix)

        ls = [np.arange(n) for n in shape]

        # Get ijk indices for points in the sub-grid
        x, y, z = [l[idx] - i for l, idx, i in zip(ls, np.where(flag), ijk)]

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
        gau_3d = [loc + ijk, vec, sgm, csgm, val]

        # Convert grid location to XYZ / coordinates
        location = [[gau_3d[0][0], gau_3d[0][1], gau_3d[0][2]]]
        location = self.lut.index2coord(location)[0]

        uncertainty = sgm * self.lut.cell_size

        return location, uncertainty

    def _splineloc(self, coa_map, win=5, upscale=10):
        """
        Fit a 3-D spline function to a region around the maximum coalescence
        in the marginalised coalescence map and interpolate by factor {upscale}
        to return a sub-grid maximum coalescence location.

        Parameters
        ----------
        coa_map : array-like
            Marginalised 3-D coalescence map.

        win : int
            Window of grid cells (+/-(win-1)//2 in x, y and z) around max
            value in coa_map to perform the fit over.

        upscale : int
            Upscaling factor to interpolate the fitted 3-D spline function by.

        Returns
        -------
        location : array-like, [x, y, z]
            Max coalescence location from spline interpolation.

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
                msg += " with maximum coalescence value"
                self.output.log(msg, self.log)

            location = self.lut.coord2grid([[mxi, myi, mzi]], inverse=True)[0]

            # Run check that spline location is within window
            if (abs(mx - mxi) > w2) or (abs(my - myi) > w2) or \
               (abs(mz - mzi) > w2):
                msg = "\t !!!! Spline error: location outside interpolation "
                msg += "window !!!!\n\t\t\tGridded Location returned"
                self.output.log(msg, self.log)

                location = self.lut.coord2grid([[mx, my, mz]], inverse=True)[0]
        else:
            msg = "\t !!!! Spline error: interpolation window crosses edge of "
            msg += "grid !!!!\n\t\t\tGridded Location returned"
            self.output.log(msg, self.log)

            location = self.lut.coord2grid([[mx, my, mz]], inverse=True)[0]

        return location

    def _calculate_location(self, map_4d):
        """
        Marginalise 4-D coalescence grid. Calculate a set of locations and
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
        self.coa_map = self.coa_map / np.nanmax(self.coa_map)

        # Fit 3-D spline function to small window around max coalescence
        # location and interpolate to determine sub-grid maximum coalescence
        # location.
        loc_spline = self._splineloc(np.copy(self.coa_map))

        # Apply Gaussian smoothing to small window around max coalescence
        # location and fit 3-D Gaussian function to determine local
        # expectation location and uncertainty
        smoothed_coa_map = self._gaufilt3d(np.copy(self.coa_map))
        loc_gau, loc_gau_err = self._gaufit3d(smoothed_coa_map, thresh=0.)

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
                                 event_uid, map_4d):
        """
        Deal with optional outputs in locate():
            plot_event_summary()
            plot_event_video()
            write_cut_waveforms()
            write_4d_coal_grid()

        Parameters
        ----------
        event_mw_data : pandas DataFrame
            Gridded maximum coa location through time across the marginal
            window. Columns = ["DT", "COA", "X", "Y", "Z"]

        event : pandas DataFrame
            Final event location information.
            Columns = ["DT", "COA", "COA_NORM", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ", "TRIG_COA", "DEC_COA",
                       "DEC_COA_NORM"]
            All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres

        out_str : str
            String {run_name}_{event_name} (figure displayed by default)

        event_uid : str
            UID of earthquake: "YYYYMMDDHHMMSSFFFF"

        map_4d : array-like
            4-D coalescence grid output from _compute()

        """

        if self.plot_event_summary or self.plot_event_video:
            quake_plot = qplot.QuakePlot(self.lut, self.data, event_mw_data,
                                         self.marginal_window, self.output.run,
                                         event, map_4d, self.coa_map)

        if self.plot_event_summary:
            timer = util.Stopwatch()
            self.output.log("\tPlotting event summary figure...", self.log)
            quake_plot.event_summary(file_str=out_str)
            self.output.log(timer(), self.log)

        if self.picker.plot_phase_picks:
            self.picker.plot(file_str=out_str, event_uid=event_uid,
                             run_path=self.output.run)

        if self.plot_event_video:
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

        try:
            del quake_plot
        except NameError:
            pass

    @property
    def sampling_rate(self):
        """Get sampling_rate"""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        """
        Set sampling_rate and try set for the onset and picker objects, if they
        exist.

        """

        try:
            self.onset.sampling_rate = value
            self.picker.sampling_rate = value
        except AttributeError:
            pass
        self._sampling_rate = value
