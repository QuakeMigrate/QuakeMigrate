# -*- coding: utf-8 -*-
"""
Module for processing waveform files stored in a data archive.

"""

import pathlib
from itertools import chain

from obspy import read, Trace, Stream, UTCDateTime
import numpy as np

import QMigrate.util as util
import QMigrate.io.quakeio as qio


class Archive(object):
    """
    Archive object

    Reads data all available data from archive between specified times. Selects
    data for requested stations to perform some clean-up and remove any gappy
    recordings.

    Attributes
    ----------
    archive_path : pathlib Path object
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE

    format : str
        File naming format of data archive

    raw_waveforms : obspy Stream object
        All raw seismic data found and read in from the archive in the
        specified time period

    signal : array-like
        Processed 3-component seismic data at the desired sampling rate only
        for desired stations with continuous data on all 3 components
        throughout the desired time period and where the data could be
        successfully resampled to the desired sampling rate

    filtered_signal : array-like
        Filtered data originally from signal

    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly
        to the desired sampling rate.

    upfactor : int, optional
        Factor by which to upsample the data (using _upsample() )to enable it
        to be decimated to the desired sampling rate.
            E.g. 40Hz -> 50Hz requires upfactor = 5.

    allstations : bool, optional
        If True, read all stations in archive for that time period. Else,
        only read specified stations.

    stations : pandas Series object
        Series object containing station names

    Methods
    -------
    path_structure(path_type="YEAR/JD/STATION")
        Set the file naming format of the data archive

    read_waveform_data(start_time, end_time, sampling_rate)
        Read in all waveform data between two times, downsample / resample if
        required to reach desired sampling rate. Return all raw data as an
        obspy Stream object and processed data for specified stations as an
        array for use by QuakeScan.

    """

    def __init__(self, station_file, archive_path, delimiter=","):
        """
        Archive object initialisation.

        Parameters
        ----------
        station_file : str
            File path to QMigrate station file: list of stations selected
            for QuakeMigrate run. Station file must contain header with
            columns: ["Latitude", "Longitude", "Elevation", "Name"]

        delimiter : char, optional
            QMigrate station file delimiter; defaults to ","

        archive_path : str
            Location of seismic data archive: e.g.: "./DATA_ARCHIVE"

        """

        self.archive_path = pathlib.Path(archive_path)

        self.format = None
        self.signal = None
        self.filtered_signal = None

        self.resample = False
        self.upfactor = None

        self.read_all_stations = False

        self.stations = qio.stations(station_file, delimiter=delimiter)["Name"]
        self.st = None

    def __str__(self):
        """
        Return short summary string of the Archive object.

        It will provide information about the archive location and structure,
        data sampling rate and time period over which the archive is being
        queried.

        """

        out = "QuakeMigrate Archive object"
        out += "\n\tArchive path\t:\t{}".format(self.archive_path)
        out += "\n\tPath structure\t:\t{}".format(self.format)
        out += "\n\tResampling\t:\t{}".format(self.resample)
        # out += "\n\tSampling rate\t:\t{}".format(self.sampling_rate)
        # out += "\n\tStart time\t:\t{}".format(str(self.start_time))
        # out += "\n\tEnd time\t:\t{}".format(str(self.end_time))
        out += "\n\tStations:"
        for station in self.stations:
            out += "\n\t\t{}".format(station)

        return out

    def path_structure(self, archive_format="YEAR/JD/STATION"):
        """
        Define the format of the data archive.

        Parameters
        ----------
        archive_format : str, optional
            Sets path type for different archive formats

        """

        if archive_format == "SeisComp3":
            self.format = "{year}/*/{station}/*/*.{station}..*.D.{year}.{jday}"
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

    def read_waveform_data(self, start_time, end_time, sampling_rate,
                           pre_pad=None, post_pad=None):
        """
        Read in the waveform data for all stations in the archive between two
        times and return station availability of the stations specified in the
        station file during this period. Downsample / resample (optional) this
        data if required to reach desired sampling rate.

        Output both processed data for stations in station file and all raw
        data in an obspy Stream object.

        Supports all formats currently supported by obspy, including: "MSEED"
        (default), "SAC", "SEGY", "GSE2" .

        Parameters
        ----------
        start_time : UTCDateTime object
            Start datetime to read waveform data

        end_time : UTCDateTime object
            End datetime to read waveform data

        sampling_rate : int
            Sampling rate in hertz

        pre_pad : float, optional
            Additional pre pad of data to cut based on user-defined pre_cut
            parameter. Defaults to none: pre_pad calculated in QuakeScan will
            be used (included in start_time).

        post_pad : float, optional
            Additional post pad of data to cut based on user-defined post_cut
            parameter. Defaults to none: post_pad calculated in QuakeScan will
            be used (included in end_time).

        """

        self.sampling_rate = sampling_rate
        self.start_time = start_time
        self.end_time = end_time

        if pre_pad is None:
            pre_pad = 0.
        if post_pad is None:
            post_pad = 0.

        samples = int(round((end_time - start_time) * sampling_rate + 1))
        files = self._load_from_path(start_time - pre_pad, end_time + post_pad)

        st = Stream()
        try:
            first = next(files)
            files = chain([first], files)
            for file in files:
                file = str(file)
                try:
                    st += read(file, starttime=start_time - pre_pad,
                               endtime=end_time + post_pad)
                except TypeError:
                    msg = "File not compatible with obspy - {}"
                    print(msg.format(file))
                    continue

            # Remove all stations with data gaps
            st.merge(method=-1)

            # Make copy of raw waveforms to output if requested, delete st
            st_raw = st.copy()
            st_selected = Stream()

            # re-populate st with only stations in station file, and only
            # data between start and end time needed for QuakeScan
            for stn in self.stations.tolist():
                st_selected += st.select(station=stn)
            st = st_selected.copy()
            for tr in st.traces:
                tr.trim(starttime=start_time, endtime=end_time)

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
            st = self._downsample(st, sampling_rate, self.upfactor)

            # Combining the data and determining station availability
            signal, availability = self._station_availability(st, samples)

        except StopIteration:
            self.availability = np.zeros(len(self.stations))
            raise util.ArchiveEmptyException

        self.raw_waveforms = st_raw
        self.signal = signal
        self.filtered_signal = np.empty((self.signal.shape))
        self.filtered_signal[:] = np.nan
        self.availability = availability

    def _station_availability(self, stream, samples):
        """
        Determine whether continuous data exists between two times for a given
        station.

        Parameters
        ----------
        stream : obspy Stream object
            Stream containing 3-component data for stations in station file

        samples : int
            Number of samples expected in the signal

        Returns
        -------
        signal : array-like
            3-component seismic data only for stations with continuous data
            on all 3 components throughout the desired time period

        availability : array-like
            Array containing 0s (no data) or 1s (data)

        """

        availability = np.zeros(len(self.stations))
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
                        channel = tr.stats.channel[-1]
                        if channel == "E" or channel == "2":
                            signal[1, i, :] = tr.data

                        if channel == "N" or channel == "1":
                            signal[0, i, :] = tr.data

                        if channel == "Z":
                            signal[2, i, :] = tr.data

        # Check to see if no traces were continuously active during this period
        if not np.any(availability):
            raise util.DataGapException

        return signal, availability

    def _load_from_path(self, start_time, end_time):
        """
        Retrieves available files between two times.

        Parameters
        ----------
        start_time : UTCDateTime object
            Start datetime to read waveform data

        end_time : UTCDateTime object
            End datetime to read waveform data

        Returns
        -------
        files : generator
            Iterator object of available waveform data files

        """

        if self.format is None:
            print("Specify the archive structure using Archive.path_structure")
            return

        dy = 0
        files = []
        start_day = UTCDateTime("{}-{}T00:00:00.0".format(start_time.year,
                                                          str(start_time.julday).zfill(3)))
        # Loop through time period by day adding files to list
        # NOTE! This assumes the archive structure is split into days.
        while start_day + (dy * 86400) <= end_time:
            now = start_time + (dy * 86400)
            if self.read_all_stations is True:
                file_format = self.format.format(year=now.year,
                                                 month=now.month,
                                                 jday=str(now.julday).zfill(3),
                                                 station="*")
                files = chain(files, self.archive_path.glob(file_format))
            else:
                for stat in self.stations.tolist():
                    file_format = self.format.format(year=now.year,
                                                 month=now.month,
                                                 jday=str(now.julday).zfill(3),
                                                 station=stat)
                    files = chain(files, self.archive_path.glob(file_format))

            dy += 1

        return files

    def _downsample(self, stream, sr, upfactor=None):
        """
        Downsample the stream to the specified sampling rate.

        Parameters
        ----------
        stream : obspy Stream object
            Contains list of Trace objects to be downsampled

        sr : int
            Output sampling rate

        Returns
        -------
        stream : obspy Stream object
            Contains list of Trace objects, with Traces downsampled / resampled
            where necessary and possible.

        """

        for trace in stream:
            if sr != trace.stats.sampling_rate:
                trace.filter("lowpass", freq=float(sr) / 2.000001, corners=2,
                             zerophase=True)
                if (trace.stats.sampling_rate % sr) == 0:
                    trace.decimate(factor=int(trace.stats.sampling_rate / sr),
                                   strict_length=False,
                                   no_filter=True)
                elif self.resample and upfactor is not None:
                    # Check the upsampled sampling rate can be decimated to sr
                    if int(trace.stats.sampling_rate * upfactor) % sr != 0:
                        raise util.BadUpfactorException
                    stream.remove(trace)
                    trace = self._upsample(trace, upfactor)
                    trace.decimate(factor=int(trace.stats.sampling_rate / sr),
                                   strict_length=False,
                                   no_filter=True)
                    stream += trace
                else:
                    msg = "Mismatched sampling rates - cannot decimate data - "
                    msg += "to resample data, set .resample = True and choose"
                    msg += " a suitable upfactor"
                    print(msg)

        return stream

    def _upsample(self, trace, upfactor):
        """
        Upsample a data stream by a given factor, prior to decimation

        Parameters
        ----------
        trace : obspy Trace object
            Trace to be upsampled
        upfactor : int
            Factor by which to upsample the data in trace

        Returns
        -------
        out : obpsy Trace object
            Upsampled trace

        """

        data = trace.data
        dnew = np.zeros(len(data) * upfactor - (upfactor - 1))
        dnew[::upfactor] = data
        for i in range(1, upfactor):
            dnew[i::upfactor] = float(i) / upfactor * data[:-1] \
                         + float(upfactor - i) / upfactor * data[1:]

        out = Trace()
        out.data = dnew
        out.stats = trace.stats
        out.stats.npts = len(out.data)
        out.stats.starttime = trace.stats.starttime
        out.stats.sampling_rate = int(upfactor * trace.stats.sampling_rate)

        return out

    @property
    def sample_size(self):
        """Get the size of a sample (units: s)"""

        return 1 / self.sampling_rate
