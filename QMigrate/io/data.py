# -*- coding: utf-8 -*-
"""
Module for processing waveform files stored in a data archive.

"""

from itertools import chain
import logging
import pathlib

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
import pandas as pd

import QMigrate.util as util


class Archive(object):
    """
    The Archive class handles the reading of archived waveform data.

    It is capable of handling any regular archive structure. Some minor cleanup
    is performed to remove waveform data from stations when there are time gaps
    greater than the sample size. There is also the option to resample (up or
    down) the waveform data by decimation. Keeps a record of which stations had
    data available for a given timestep.

    Parameters
    ----------
    stations : `pandas.DataFrame` object
        Station information.
        Columns ["Latitude", "Longitude", "Elevation", "Name"]
    archive_path : str
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE.
    kwargs : **dict
        See Archive Attributes for details.

    Attributes
    ----------
    archive_path : `pathlib.Path` object
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE.
    format : str
        File naming format of data archive.
    read_all_stations : bool, optional
        If True, read all stations in archive for that time period. Else, only
        read specified stations.
    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly
        to the desired sampling rate.
    stations : `pandas.Series` object
        Series object containing station names.
    upfactor : int, optional
        Factor by which to upsample the data (using _upsample() )to enable it
        to be decimated to the desired sampling rate, e.g. 40Hz -> 50Hz
        requires upfactor = 5.

    Methods
    -------
    path_structure(path_type="YEAR/JD/STATION")
        Set the file naming format of the data archive.
    read_waveform_data(starttime, endtime, sampling_rate)
        Read in all waveform data between two times, downsample / resample if
        required to reach desired sampling rate. Return all raw data as an
        obspy Stream object and processed data for specified stations as an
        array for use by QuakeScan.

    """

    def __init__(self, stations, archive_path, **kwargs):
        """Instantiate the Archive object."""

        self.archive_path = pathlib.Path(archive_path)
        self.stations = stations["Name"]

        self.format = kwargs.get("format", "")
        self.read_all_stations = kwargs.get("read_all_stations", False)
        self.resample = kwargs.get("resample", False)
        self.sampling_rate = kwargs.get("sampling_rate")
        self.upfactor = kwargs.get("upfactor")

    def __str__(self):
        """Returns a short summary string of the Archive object."""

        out = ("QuakeMigrate Archive object"
               f"\n\tArchive path\t:\t{self.archive_path}"
               f"\n\tPath structure\t:\t{self.format}"
               f"\n\tResampling\t:\t{self.resample}"
               "\n\tStations:")
        for station in self.stations:
            out += f"\n\t\t{station}"

        return out

    def path_structure(self, archive_format="YEAR/JD/STATION"):
        """
        Define the path structure of the data archive.

        Parameters
        ----------
        archive_format : str, optional
            Sets path type for different archive formats.

        """

        if archive_format == "SeisComp3":
            self.format = "{year}/*/{station}/BH*/*.{station}.*.*.D.{year}.{jday}"
        elif archive_format == "YEAR/JD/*_STATION_*":
            self.format = "{year}/{jday}/*_{station}_*"
        elif archive_format == "YEAR/JD/STATION":
            self.format = "{year}/{jday}/{station}*"
        elif archive_format == "STATION.YEAR.JULIANDAY":
            self.format = "*{station}.*.{year}.{jday}"
        elif archive_format == "/STATION/STATION.YearMonthDay":
            self.format = "{station}/{station}.{year}{month:02d}{day:02d}"
        elif archive_format == "YEAR_JD/STATION*":
            self.format = "{year}_{jday}/{station}*"
        elif archive_format == "YEAR_JD/STATION_*":
            self.format = "{year}_{jday}/{station}_*"

    def read_waveform_data(self, starttime, endtime, pre_pad=0., post_pad=0.):
        """
        Read in the waveform data for all stations in the archive between two
        times and return station availability of the stations specified in the
        station file during this period. Downsample / resample (optional) this
        data if required to reach desired sampling rate.

        Output both processed data for stations in station file and all raw
        data in an obspy Stream object.

        Supports all formats currently supported by ObsPy, including: "MSEED"
        (default), "SAC", "SEGY", "GSE2" .

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object, optional
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object, optional
            Timestamp up to which to read waveform data.
        pre_pad : float, optional
            Additional pre pad of data to cut based on user-defined pre_cut
            parameter. Defaults to none: pre_pad calculated in QuakeScan will
            be used (included in starttime).
        post_pad : float, optional
            Additional post pad of data to cut based on user-defined post_cut
            parameter. Defaults to none: post_pad calculated in QuakeScan will
            be used (included in endtime).

        Returns
        -------
        data : `QMigrate.io.data.SignalData` object
            Object containing the archived data that satisfies the query.

        """

        data = SignalData(starttime=starttime, endtime=endtime,
                          sampling_rate=self.sampling_rate)
        data.stations = self.stations

        samples = int(round((endtime - starttime) * self.sampling_rate + 1))
        files = self._load_from_path(starttime - pre_pad, endtime + post_pad)

        st = Stream()
        try:
            first = next(files)
            files = chain([first], files)
            for file in files:
                file = str(file)
                try:
                    st += read(file, starttime=starttime - pre_pad,
                               endtime=endtime + post_pad)
                except TypeError:
                    logging.info(f"File not compatible with ObsPy - {file}")
                    continue

            # Merge all traces with contiguous data, or overlapping data which
            # exactly matches (== st._cleanup(); i.e. no clobber)
            st.merge(method=-1)

            # Make copy of raw waveforms to output if requested
            data.raw_waveforms = st.copy()
            st_selected = Stream()

            # Re-populate st with only stations in station file, and only
            # data between start and end time needed for QuakeScan
            for station in self.stations:
                st_selected += st.select(station=station)
            st = st_selected.copy()
            for tr in st:
                tr.trim(starttime=starttime, endtime=endtime)
                if not bool(tr):
                    st.remove(tr)

            # Remove stations which have gaps in at least one trace
            gaps = st.get_gaps()
            if gaps:
                stations = np.unique(np.array(gaps)[:, 1]).tolist()
                for station in stations:
                    traces = st.select(station=station)
                    for trace in traces:
                        st.remove(trace)

            # Test if the stream is completely empty
            # (see __nonzero__ for obspy Stream object)
            if not bool(st):
                self.availability = np.zeros(len(self.stations))
                raise util.DataGapException

            # Detrend and downsample / resample stream if required
            st.detrend("linear")
            st.detrend("demean")
            st = self._downsample(st, self.sampling_rate, self.upfactor)

            # Combining the data and determining station availability
            data.signal, availability = self._station_availability(st, samples)

        except StopIteration:
            self.availability = np.zeros(len(self.stations))
            raise util.ArchiveEmptyException

        self.availability = availability

        return data

    def _station_availability(self, stream, samples):
        """
        Determine whether continuous data exists between two times for a given
        station.

        Parameters
        ----------
        stream : `obspy.Stream` object
            Stream containing 3-component data for stations in station file.
        samples : int
            Number of samples expected in the signal.

        Returns
        -------
        signal : `numpy.ndarray`, shape(3, nstations, nsamples)
            3-component seismic data only for stations with continuous data
            on all 3 components throughout the desired time period.
        availability : `np.ndarray` of ints, shape(nstations)
            Array containing 0s (no data) or 1s (data).

        """

        availability = np.zeros(len(self.stations)).astype(int)
        signal = np.zeros((3, len(self.stations), int(samples)))

        for i, station in enumerate(self.stations):
            tmp_st = stream.select(station=station)
            if len(tmp_st) == 3:
                if (tmp_st[0].stats.npts == samples and
                        tmp_st[1].stats.npts == samples and
                        tmp_st[2].stats.npts == samples):

                    # Defining the station as available
                    availability[i] = 1

                    for tr in tmp_st:
                        # Check channel name has 3 characters
                        try:
                            channel = tr.stats.channel[2]
                            # Assign data to signal array by component
                            if channel == "E" or channel == "2":
                                signal[1, i, :] = tr.data
                            elif channel == "N" or channel == "1":
                                signal[0, i, :] = tr.data
                            elif channel == "Z":
                                signal[2, i, :] = tr.data
                            else:
                                raise util.ChannelNameException(tr)

                        except IndexError:
                            raise util.ChannelNameException(tr)

        # Check to see if no traces were continuously active during this period
        if not np.any(availability):
            self.availability = availability
            raise util.DataGapException

        return signal, availability

    def _load_from_path(self, starttime, endtime):
        """
        Retrieves available files between two times.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to read waveform data.

        Returns
        -------
        files : generator
            Iterator object of available waveform data files.

        """

        if self.format is None:
            logging.info("No archive structure specified - set with "
                         " Archive.path_structure = ")
            return

        start_day = UTCDateTime(starttime.date)

        dy = 0
        files = []
        # Loop through time period by day adding files to list
        # NOTE! This assumes the archive structure is split into days.
        while start_day + (dy * 86400) <= endtime:
            now = starttime + (dy * 86400)
            if self.read_all_stations is True:
                file_format = self.format.format(year=now.year,
                                                 month=f"{now.month:02d}",
                                                 day=f"{now.day:02d}",
                                                 jday=f"{now.julday:03d}",
                                                 station="*")
                files = chain(files, self.archive_path.glob(file_format))
            else:
                for station in self.stations:
                    file_format = self.format.format(year=now.year,
                                                     month=f"{now.month:02d}",
                                                     day=f"{now.day:02d}",
                                                     jday=f"{now.julday:03d}",
                                                     station=station)
                    files = chain(files, self.archive_path.glob(file_format))
            dy += 1

        return files

    def _downsample(self, stream, sr, upfactor=None):
        """
        Downsample the stream to the specified sampling rate.

        Parameters
        ----------
        stream : `obspy.Stream` object
            Contains list of Trace objects to be downsampled.
        sr : int
            Output sampling rate.

        Returns
        -------
        stream : `obspy.Stream` object
            Contains list of `obspy.Trace` objects, downsampled / resampled
            where necessary and possible.

        """

        sr = self.sampling_rate
        for trace in stream:
            if sr != trace.stats.sampling_rate:
                if (trace.stats.sampling_rate % sr) == 0:
                    trace.filter("lowpass", freq=float(sr) / 2.000001,
                                 corners=2, zerophase=True)
                    trace.decimate(factor=int(trace.stats.sampling_rate / sr),
                                   strict_length=False,
                                   no_filter=True)
                elif self.resample and upfactor is not None:
                    # Check the upsampled sampling rate can be decimated to sr
                    if int(trace.stats.sampling_rate * upfactor) % sr != 0:
                        raise util.BadUpfactorException(trace)
                    stream.remove(trace)
                    trace = self._upsample(trace, upfactor)
                    trace.filter("lowpass", freq=float(sr) / 2.000001,
                                 corners=2, zerophase=True)
                    trace.decimate(factor=int(trace.stats.sampling_rate / sr),
                                   strict_length=False,
                                   no_filter=True)
                    stream += trace
                else:
                    logging.info("Mismatched sampling rates - cannot decimate "
                                 "data - to resample data, set .resample "
                                 "= True and choose a suitable upfactor")

        return stream

    def _upsample(self, trace, upfactor):
        """
        Upsample a data stream by a given factor, prior to decimation. The
        upsampling is done using a linear interpolation.

        Parameters
        ----------
        trace : `obspy.Trace` object
            Trace to be upsampled.
        upfactor : int
            Factor by which to upsample the data in trace.

        Returns
        -------
        out : `obpsy.Trace` object
            Upsampled trace.

        """

        data = trace.data
        dnew = np.zeros(len(data)*upfactor - (upfactor - 1))
        dnew[::upfactor] = data
        for i in range(1, upfactor):
            dnew[i::upfactor] = float(i)/upfactor*data[1:] \
                         + float(upfactor - i)/upfactor*data[:-1]

        out = Trace()
        out.data = dnew
        out.stats = trace.stats
        out.stats.npts = len(out.data)
        out.stats.starttime = trace.stats.starttime
        out.stats.sampling_rate = int(upfactor * trace.stats.sampling_rate)

        return out


class SignalData:
    """
    The SignalData class encapsulates the signal data to be returned from an
    Archive query.

    Parameters
    ----------
    starttime : `obspy.UTCDateTime` object
        Timestamp of first sample of waveform data.
    endtime : `obspy.UTCDateTime` object
        Timestamp of last sample of waveform data.
    sampling_rate : int
        Sampling rate of waveform data.

    Attributes
    ----------
    filtered_signal : `numpy.ndarray`, shape(3, nstations, nsamples)
        Filtered data originally from signal.
    raw_waveforms : `obspy.Stream` object
        All raw seismic data found and read in from the archive in the
        specified time period.
    signal : `numpy.ndarray`, shape(3, nstations, nsamples)
        Processed 3-component seismic data at the desired sampling rate only
        for desired stations with continuous data on all 3 components
        throughout the desired time period and where the data could be
        successfully resampled to the desired sampling rate.

    Methods
    -------
    times
        Utility function to generate the corresponding timestamps for the
        waveform and coalescence data.

    """

    def __init__(self, starttime, endtime, sampling_rate):
        """Instantiate the SignalData object."""

        self.starttime = starttime
        self.endtime = endtime
        self.sampling_rate = sampling_rate

    def times(self, **kwargs):
        """
        Utility function to generate timestamps between `data.starttime` and
        `data.endtime`, with a sample size of `data.sample_size`

        Returns
        -------
        times : `numpy.ndarray`, shape(nsamples)
            Timestamps for the timeseries data.

        """

        # Utilise the .times() method of `obspy.Trace` objects
        tr = Trace(header={"npts": self.signal.shape[-1],
                           "sampling_rate": self.sampling_rate,
                           "starttime": self.starttime})
        return tr.times(**kwargs)

    @property
    def sample_size(self):
        """Get the size of a sample (units: s)."""

        return 1 / self.sampling_rate


EVENT_FILE_COLS = ["DT", "COA", "COA_NORM", "X", "Y", "Z",
                   "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                   "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                   "LocalGaussian_ErrZ", "GlobalCovariance_X",
                   "GlobalCovariance_Y", "GlobalCovariance_Z",
                   "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                   "GlobalCovariance_ErrZ", "TRIG_COA", "DEC_COA",
                   "DEC_COA_NORM", "ML", "ML_Err"]


class Event:
    """
    Light class to encapsulate information about an event, including waveform
    data (raw, filtered, unfiltered), locations, magnitudes, origin times,
    coalescence information.

    Parameters
    ----------
    triggered_event : `pandas.Series` object
        Contains information on the event output by the trigger stage.

    Attributes
    ----------

    """

    def __init__(self, triggered_event):
        """Instantiate the Event object."""

        self.coa_time = triggered_event["CoaTime"]
        self.uid = triggered_event["EventID"]

        try:
            self.trigger_info = {"TRIG_COA": triggered_event["COA_V"],
                                 "DEC_COA": triggered_event["COA"],
                                 "DEC_COA_NORM": triggered_event["COA_NORM"]}
        except KeyError:
            # --- Backwards compatibility ---
            self.trigger_info = {"TRIG_COA": triggered_event["COA_V"],
                                 "DEC_COA": np.nan,
                                 "DEC_COA_NORM": np.nan}

        self.locations = {}
        self.gaus = {}
        self.wins = {}

    def add_coalescence(self, times, max_coa, max_coa_n, coord, map4d):
        """
        Append output of _compute from locate() to `obspy.Stream` object.

        Parameters
        ----------
        times : `numpy.ndarray` of `obspy.UTCDateTime` objects, shape(nsamples)
            Timestamp of first sample of coalescence data.
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

        self.coa_data = pd.DataFrame(np.array((times, max_coa, max_coa_n,
                                               coord[:, 0], coord[:, 1],
                                               coord[:, 2])).transpose(),
                                     columns=["DT", "COA", "COA_NORM",
                                              "X", "Y", "Z"])
        self.map4d = map4d

    def add_covariance_location(self, xyz, xyz_unc):
        """
        Add the location determined by calculating the 3-D covariance of the
        marginalised coalescence map filtered above a percentile threshold.

        Parameters
        ----------
        xyz : `numpy.ndarray' of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of covariance location.
        xyz_unc : `numpy.ndarray' of floats, shape(3)
            One sigma uncertainties on the covariance location (in m).

        """

        self.locations["covariance"] = {"GlobalCovariance_X": xyz[0],
                                        "GlobalCovariance_Y": xyz[1],
                                        "GlobalCovariance_Z": xyz[2],
                                        "GlobalCovariance_ErrX": xyz_unc[0],
                                        "GlobalCovariance_ErrY": xyz_unc[1],
                                        "GlobalCovariance_ErrZ": xyz_unc[2]}

    def add_gaussian_location(self, xyz, xyz_unc):
        """
        Add the location determined by fitting a 3-D Gaussian to a small window
        around the Gaussian smoothed maximum coalescence location.

        Parameters
        ----------
        xyz : `numpy.ndarray' of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of Gaussian location.
        xyz_unc : `numpy.ndarray' of floats, shape(3)
            One sigma uncertainties on the Gaussian location (in m).

        """

        self.locations["gaussian"] = {"LocalGaussian_X": xyz[0],
                                      "LocalGaussian_Y": xyz[1],
                                      "LocalGaussian_Z": xyz[2],
                                      "LocalGaussian_ErrX": xyz_unc[0],
                                      "LocalGaussian_ErrY": xyz_unc[1],
                                      "LocalGaussian_ErrZ": xyz_unc[2]}

    def add_spline_location(self, xyz):
        """
        Add the location determined by fitting a 3-D spline to a small window
        around the maximum coalescence location and interpolating.

        Parameters
        ----------
        xyz : `numpy.ndarray' of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of best-fitting location.

        """

        self.locations["spline"] = dict(zip(["X", "Y", "Z"], xyz))

    def trim2window(self, marginal_window):
        """
        Trim the coalescence data to be within the marginal window.

        Parameters
        ----------
        marginal_window : float
            Half-width of window centred on the maximum coalescence time.

        """

        window_start = self.otime - marginal_window
        window_end = self.otime + marginal_window

        self.coa_data = self.coa_data[(self.coa_data["DT"] >= window_start) &
                                      (self.coa_data["DT"] <= window_end)]
        self.map4d = self.map4d[:, :, :,
                                self.coa_data.index[0]:self.coa_data.index[-1]]
        self.coa_data.reset_index(drop=True, inplace=True)

    def in_marginal_window(self, marginal_window):
        """
        Test if triggered event time is within marginal window around
        the maximum coalescence time (origin time).

        Parameters
        ----------
        marginal_window : float
            Half-width of window centred on the maximum coalescence time.

        Returns
        -------
        cond : bool
            Result of test.

        """

        window_start = self.otime - marginal_window
        window_end = self.otime + marginal_window
        cond = (self.coa_time > window_start) or (self.coa_time < window_end)
        if not cond:
            logging.info(f"\tEvent {self.uid} is outside marginal window.")
            logging.info("\tDefine more realistic error - the marginal "
                         "window should be an estimate of overall uncertainty")
            logging.info("\tdetermined from expected spatial uncertainty"
                         " and uncertainty in the seismic velocity model.\n")
            logging.info(util.log_spacer)

        return cond

    def write(self, run):
        """
        Write event. to a .event file.

        Parameters
        ----------
        run : `QMigrate.io.Run` object
            Light class encapsulating i/o path information for a given run.

        """

        fpath = run.path / "locate" / run.subname / "events"
        fpath.mkdir(exist_ok=True, parents=True)

        # --- Will be moved when magnitudes added ---
        magnitudes = {"ML": np.nan,
                      "ML_Err": np.nan}
        # --- Will be moved when magnitudes added ---

        out = {**self.trigger_info, **magnitudes}
        out = {**out, **self.max_coalescence}
        for key, location in self.locations.items():
            out = {**out, **location}

        self.df = pd.DataFrame([out])[EVENT_FILE_COLS]

        fstem = f"{self.uid}"
        file = (fpath / fstem).with_suffix(".event")
        self.df.to_csv(file, index=False)

    @property
    def hypocentre(self):
        """Get the hypocentral location based on the peak coalescence."""
        idxmax = self.coa_data["COA"].astype(float).idxmax()
        return self.coa_data.iloc[idxmax][["X", "Y", "Z"]].values

    @property
    def max_coalescence(self):
        """Get information related to the maximum coalescence."""
        idxmax = self.coa_data["COA"].astype("float").idxmax()
        max_coa = self.coa_data.iloc[idxmax]
        keys = ["DT", "COA", "COA_NORM"]

        return dict(zip(keys, max_coa[keys].values))

    @property
    def otime(self):
        """Get the origin time based on the peak coalescence."""
        idxmax = self.coa_data["COA"].astype(float).idxmax()
        return self.coa_data.iloc[idxmax]["DT"]
