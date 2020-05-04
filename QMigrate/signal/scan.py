# -*- coding: utf-8 -*-
"""
Module to perform core QuakeMigrate functions: detect() and locate().

"""

import warnings

import numpy as np
from obspy import UTCDateTime
import pandas as pd
from scipy.interpolate import Rbf
from scipy.signal import fftconvolve

from QMigrate.core import find_max_coa, migrate
from QMigrate.io import (Event, Run, ScanmSEED, read_triggered_events,
                         write_availability, write_cut_waveforms)
from .onset import Onset
from .pick import GaussianPicker, PhasePicker
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

    def __init__(self, archive, lut, onset, run_path, run_name,
                 log=False, **kwargs):
        """Instantiate the QuakeScan object."""

        super().__init__()

        self.archive = archive
        self.lut = lut
        if isinstance(onset, Onset):
            self.onset = onset
        else:
            raise util.OnsetTypeError
        self.onset.post_pad = lut.max_ttime

        # --- Set up i/o ---
        self.run = Run(run_path, run_name, kwargs.get("run_subname", ""))
        self.log = kwargs.get("log", True)

        picker = kwargs.get("picker")
        if picker is None:
            self.picker = GaussianPicker(onset=onset)
        elif isinstance(picker, PhasePicker):
            self.picker = picker
        else:
            raise util.PickerTypeError

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
        """Return short summary string of the QuakeScan object."""

        return ("\tScan parameters:\n"
                f"\t\tData sampling rate = {self.sampling_rate} Hz\n"
                f"\t\tTime step = {self.timestep} s\n"
                f"\t\tMarginal window = {self.marginal_window} s\n"
                f"\t\tThread count = {self.threads}\n")

    def detect(self, starttime, endtime):
        """
        Scans through continuous data calculating coalescence on a (decimated)
        3-D grid by back-migrating onset (characteristic) functions.

        Parameters
        ----------
        starttime : str
            Timestamp from which to run continuous scan (detect).
        endtime : str
            Timestamp up to which to run continuous scan (detect).
            Note: the last sample returned will be that which immediately
            precedes this timestamp.

        """

        # Configure logging
        self.run.stage = "detect"

        starttime, endtime = UTCDateTime(starttime), UTCDateTime(endtime)
        if starttime > endtime:
            raise util.TimeSpanException

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
        msg = msg.format(str(starttime), str(endtime), self.time_step,
                         self.n_cores, self.sampling_rate)
        self.output.log(msg, self.log)

        self._continuous_compute(starttime, endtime)

    def locate(self, starttime=None, endtime=None, trigger_file=None):
        """
        Re-computes the 3-D coalescence on an undecimated grid for a short
        time window around each candidate earthquake triggered from the
        (decimated) continuous detect scan. Calculates event location and
        uncertainties, makes phase arrival picks, plus multiple optional
        plotting / data outputs for further analysis and processing.

        Parameters
        ----------
        starttime : str, optional
            Timestamp from which to include events in the locate scan.
        endtime : str, optional
            Timestamp up to which to include events in the locate scan.
        trigger_file : str, optional
            File containing triggered events to be located.

        """

        # Configure logging
        self.run.stage = "locate"

        if not (starttime is None and endtime is None):
            starttime, endtime = UTCDateTime(starttime), UTCDateTime(endtime)
            if starttime > endtime:
                raise util.TimeSpanException
        if trigger_file is None and starttime is None and endtime is None:
            raise RuntimeError("Must supply an input argument.")
        if (starttime is None) ^ (endtime is None):
            raise RuntimeError("Must supply a starttime AND an endtime.")

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
        msg = msg.format(str(starttime), str(endtime), self.n_cores)
        self.output.log(msg, self.log)

        if trigger_file is not None:
            self._locate_events(trigger_file=trigger_file)
        else:
            self._locate_events(starttime=starttime, endtime=endtime)

    def _continuous_compute(self, starttime, endtime):
        """
        Compute coalescence between two timestamps, divided into increments of
        `timestep`. Outputs coalescence and station availability data to file.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to compute continuous coalescence.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to compute continuous compute.
            Note: the last sample returned will be that which immediately
            precedes this timestamp.

        """

        coalescence = ScanmSEED(self.run, self.continuous_scanmseed_write,
                                self.sampling_rate)

        t_length = self.pre_pad + self.post_pad + self.time_step
        self.pre_pad += np.ceil(t_length * 0.06)
        self.pre_pad = int(np.ceil(self.pre_pad * self.sampling_rate)
                           / self.sampling_rate * 1000) / 1000
        self.post_pad += np.ceil(t_length * 0.06)
        self.post_pad = int(np.ceil(self.post_pad * self.sampling_rate)
                            / self.sampling_rate * 1000) / 1000

        nsteps = int(np.ceil((endtime - starttime) / self.timestep))
        availability = pd.DataFrame(index=np.arange(nsteps),
                                    columns=self.lut.maps.keys())

        for i in range(nsteps):
            w_beg = starttime + self.timestep*i - self.pre_pad
            w_end = starttime + self.timestep*(i + 1) + self.post_pad
            msg = ("~" * 19) + " Processing : {} - {} " + ("~" * 19)
            msg = msg.format(str(w_beg), str(w_end))
            self.output.log(msg, self.log)

            try:
                data = self.archive.read_waveform_data(w_beg, w_end)
                coalescence.append(*self._compute(data))
            except util.ArchiveEmptyException as e:
                msg = "!" * 24 + " " * 16
                msg += " No files in archive for this time step "
                msg += " " * 16 + "!" * 24
                self.output.log(msg, self.log)
                coalescence.empty(starttime, self.timestep, i, e.msg)
            except util.DataGapException as e:
                msg = "!" * 24 + " " * 9
                msg += "All available data for this time period contains gaps"
                msg += " " * 10 + "!" * 24
                msg += "\n" + "!" * 24 + " " * 11
                msg += "or data not available at start/end of time period"
                msg += " " * 12 + "!" * 24
                self.output.log(msg, self.log)
                coalescence.empty(starttime, self.timestep, i, e.msg)

            availability.loc[i] = self.archive.availability
            availability.rename(index={i: str(starttime + self.timestep*i)},
                                inplace=True)

        if not coalescence.written:
            coalescence.write()
        write_availability(self.run, availability)

    def _locate_events(self, **kwargs):
        """
        Loop through list of earthquakes read in from trigger results and
        re-compute coalescence; output phase picks, event location and
        uncertainty, plus optional plots and outputs.

        Parameters
        ----------
        kwargs : **dict
            Can contain:
            starttime : `obspy.UTCDateTime` object, optional
                Timestamp from which to include events in the locate scan.
            endtime : `obspy.UTCDateTime` object, optional
                Timestamp up to which to include events in the locate scan.
            trigger_file : str, optional
                File containing triggered events to be located.

        """

        triggered_events = read_triggered_events(self.run, **kwargs)
        n_events = len(triggered_events.index)

        # Adjust pre- and post-pad to take into account cosine taper
        t_length = self.pre_pad + 4*self.marginal_window + self.post_pad
        self.pre_pad += np.ceil(t_length * 0.06)
        self.pre_pad = int(np.ceil(self.pre_pad * self.sampling_rate)
                           / self.sampling_rate * 1000) / 1000
        self.post_pad += np.ceil(t_length * 0.06)
        self.post_pad = int(np.ceil(self.post_pad * self.sampling_rate)
                            / self.sampling_rate * 1000) / 1000

        for i, triggered_event in triggered_events.iterrows():
            event = Event(triggered_event)
            w_beg = event.coa_time - 2*self.marginal_window - self.pre_pad
            w_end = event.coa_time + 2*self.marginal_window + self.post_pad
            msg = ("=" * 110 + f"\n\tEVENT - {i+1} of {n_events} - {event.uid}\n"
                   + "=" * 110 + "\n\n\tDetermining event location...\n")
            self.output.log(msg, self.log)

            timer = util.Stopwatch()
            self.output.log("\tReading waveform data...", self.log)
            try:
                self._read_event_waveform_data(w_beg, w_end)
                event.add_coalescence(*self._compute(event.data))
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

            # --- Trim coalescence map to marginal window ---
            if event.in_marginal_window(self.marginal_window):
                event.trim2window(self.marginal_window)
            else:
                del event
                continue

            # Determining earthquake location error
            timer = util.Stopwatch()
            self.output.log(("\tDetermining earthquake location and "
                             "uncertainty..."), self.log)
            marginalised_coalescence = self._calculate_location(event)
            self.output.log(timer(), self.log)

            self.output.log("\tMaking phase picks...", self.log)
            self.picker.pick_phases(self.data, self.lut, event.max_coalescence,
                                    event.uid, self.output)

            event.write(self.run)

            if self.plot_event_summary:
                event_summary(self.run, event, marginalised_coalescence,
                              self.lut)

            if self.plot_event_video:
                logging.info("Support for event videos coming soon.")

            if self.write_cut_waveforms:
                write_cut_waveforms(self.run, event, self.cut_waveform_format)

            self.output.log("=" * 110 + "\n", self.log)

            del event, marginalised_coalescence

    def _compute(self, data):
        """
        Compute 3-D coalescence between two time stamps.

        Parameters
        ----------
        data : `QMigrate.io.data.SignalData` object
            Light class encapsulating data returned by an archive query.

        Returns
        -------
        times : `numpy.ndarray` of `obspy.UTCDateTime` objects, shape(nsamples)
            Timestamps for the coalescence data.
        max_coa : `numpy.ndarray` of floats, shape(nsamples)
            Coalescence value through time.
        max_coa_n : `numpy.ndarray` of floats, shape(nsamples)
            Normalised coalescence value through time.
        coord : `numpy.ndarray` of floats, shape(nsamples)
            Location of maximum coalescence through time in input projection
            space.
        map4d : `numpy.ndarry`, shape(nx, ny, nz, nsamp), optional
            4-D coalescence map.

        """

        # --- Calculate continuous coalescence within 3-D volume ---
        onsets = self.onset.calculate_onsets(data)
        ttimes = self.lut.ttimes(self.sampling_rate)
        fsmp = util.time2sample(self.pre_pad, self.sampling_rate)
        lsmp = util.time2sample(self.post_pad, self.sampling_rate)
        avail = np.sum(self.archive.availability)*2
        map4d = migrate(onsets, ttimes, fsmp, lsmp, avail, self.threads)

        # --- Find continuous peak coalescence in 3-D volume ---
        max_coa, max_coa_n, max_idx = find_max_coa(map4d, self.threads)
        coord = self.lut.index2coord(max_idx, unravel=True)

        if self.run.stage == "detect":
            del map4d
            return data.starttime, max_coa, max_coa_n, coord
        else:
            times = data.times(type="utcdatetime")
            return times, max_coa, max_coa_n, coord, map4d

    def _read_event_waveform_data(self, w_beg, w_end):
        """
        Read waveform data for a triggered event.

        Parameters
        ----------
        w_beg : `obpsy.UTCDateTime` object
            Timestamp from which to read waveform data.
        w_end : `obspy.UTCDateTime` object
            Timestamp up to which to read waveform data.

        Returns
        -------
        data : `QMigrate.io.data.SignalData` object
            Light class encapsulating data returned by an archive query.

        """

        # Extra pre- and post-pad default to None
        pre_pad = post_pad = 0.
        if self.pre_cut:
            # only subtract 1*marginal_window so if the event otime moves by
            # this much the selected pre_cut can still be applied
            pre_pad = self.pre_cut - self.marginal_window - self.pre_pad
            if pre_pad < 0:
                msg = (f"\t\tWarning: specified pre_cut {self.pre_cut} is"
                       "shorter than default pre_pad\n"
                       f"\t\t\tCutting from pre_pad = {self.pre_pad}")
                logging.info(msg)
                pre_pad = 0.

        if self.post_cut:
            # only subtract 1*marginal_window so if the event otime moves by
            # this much the selected post_cut can still be applied
            post_pad = self.post_cut - self.marginal_window - \
                       self.post_pad
            if post_pad < 0:
                msg = (f"\t\tWarning: specified post_cut {self.post_cut} is"
                       " shorter than default post_pad\n"
                       f"t\t\tCutting to post_pad = {self.post_pad}")
                logging.info(msg)
                post_pad = 0.

        return self.archive.read_waveform_data(w_beg, w_end, pre_pad, post_pad)

    def _calculate_location(self, event):
        """
        Marginalise the 4-D coalescence grid and calculate a set of locations
        and associated uncertainties by:
            (1) calculating the covariance of the entire coalescence map;
            (2) smoothing and fitting a 3-D Gaussian function and ..
            (3) fitting a 3-D spline function ..
                to a region around the maximum coalescence location in the
                marginalised 3-D coalescence map.

        Parameters
        ----------
        event : `QMigrate.io.Event` object
            Light class encapsulating signal, onset, and location information
            for a given event.

        Returns
        -------
        coa_map : array-like
            Marginalised 3-D coalescence map.

        """

        # --- Marginalise and normalise the coalescence grid ---
        coa_map = np.sum(event.map4d, axis=-1)
        coa_map = coa_map / np.nanmax(coa_map)

        # --- Determine best-fitting interpolated spline location ---
        event.add_spline_location(self._splineloc(np.copy(coa_map)))

        # --- Determine best-fitting Gaussian location and uncertainty ---
        smoothed_coa_map = self._gaufilt3d(np.copy(coa_map))
        event.add_gaussian_location(*self._gaufit3d(smoothed_coa_map))

        # --- Determine global covariance location and uncertainty ---
        event.add_covariance_location(*self._covfit3d(np.copy(coa_map)))

        return coa_map

    def _splineloc(self, coa_map, win=5, upscale=10):
        """
        Fit a 3-D spline function to a region around the maximum coalescence
        in the marginalised coalescence map and interpolate by factor `upscale`
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
            logging.info(f"\t\tGridded loc: {mx}   {my}   {mz}")
            logging.info(f"\t\tSpline  loc: {mxi} {myi} {mzi}")

            # Run check that spline location is within grid-cell
            if (abs(mx - mxi) > 1) or (abs(my - myi) > 1) or \
               (abs(mz - mzi) > 1):
                logging.info("\tSpline warning: spline location outside grid "
                             "cell with maximum coalescence value")

            location = self.lut.index2coord([[mxi, myi, mzi]])[0]

            # Run check that spline location is within window
            if (abs(mx - mxi) > w2) or (abs(my - myi) > w2) or \
               (abs(mz - mzi) > w2):
                logging.info("\t !!!! Spline error: location outside"
                             "interpolation window !!!!")
                logging.info("\t\t\tGridded Location returned")

                location = self.lut.index2coord([[mx, my, mz]])[0]
        else:
            logging.info("\t !!!! Spline error: interpolation window crosses "
                         "edge of grid !!!!")
            logging.info("\t\t\tGridded Location returned")

            location = self.lut.index2coord([[mx, my, mz]])[0]

        return location

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

        # Get shape of 3-D coalescence map and max coalescence grid location
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

    def _covfit3d(self, coa_map, thresh=0.88, win=None):
        """
        Calculate the 3-D covariance of the marginalised coalescence map,
        filtered above a percentile threshold `thresh`. Optionally can also
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

    def _gaufilt3d(self, map3d, sgm=0.8, shp=None):
        """
        Smooth the 3-D marginalised coalescence map using a 3-D Gaussian
        function to enable a better Gaussian fit to the data to be calculated.

        Parameters
        ----------
        map3d : array-like
            Marginalised 3-D coalescence map.
        sgm : float
            Sigma value (in grid cells) for the 3-D Gaussian filter function;
            bigger sigma leads to more aggressive (long wavelength) smoothing.
        shp : array-like, optional
            Shape of volume.

        Returns
        -------
        smoothed_map : array-like
            Gaussian smoothed 3-D coalescence map.

        """

        if shp is None:
            shp = map3d.shape

        # Construct 3-D Gaussian filter
        flt = util.gaussian_3d(*shp, sgm)
        # Convolve map_3d and 3-D Gaussian filter
        smoothed_map = fftconvolve(map3d, flt, mode="same")
        # Mirror and convolve again (to avoid "phase-shift")
        smoothed_map = smoothed_map[::-1, ::-1, ::-1] / np.nanmax(smoothed_map)
        smoothed_map = fftconvolve(smoothed_map, flt, mode="same")
        # Final mirror and normalise
        smoothed_map = smoothed_map[::-1, ::-1, ::-1] / np.nanmax(smoothed_map)

        return smoothed_map

    def _mask3d(self, n, i, window):
        """
        Creates a mask that can be applied to a 3-D grid.

        Parameters
        ----------
        n : array-like, int
            Shape of grid.
        i : array-like, int
            Location of cell around which to mask.
        window : int
            Size of window around cell to mask - window of grid cells is
            +/-(win-1)//2 in x, y and z.

        Returns
        -------
        mask : array-like
            Masking array.

        """

        n = np.array(n)
        i = np.array(i)

        w2 = (window - 1) // 2

        x1, y1, z1 = np.clip(i - w2, 0 * n, n)
        x2, y2, z2 = np.clip(i + w2 + 1, 0 * n, n)

        mask = np.zeros(n, dtype=np.bool)
        mask[x1:x2, y1:y2, z1:z2] = True

        return mask

    @property
    def sampling_rate(self):
        """Get sampling_rate"""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        """
        Set sampling_rate and try distribute to other objects.

        """

        try:
            self.archive.sampling_rate = value
            self.onset.sampling_rate = value
            self.picker.sampling_rate = value
        except AttributeError:
            pass
        self._sampling_rate = value
