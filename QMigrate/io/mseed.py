# -*- coding: utf-8 -*-
"""
Module for processing mSEED files.

Provides information on station in a given archive.

"""

import pathlib
from itertools import chain

import obspy
import numpy as np

import QMigrate.core.model as cmod


class MSEED(object):
    """
    mSEED object

    Reads data from archive, performs some clean-up and removes any gappy
    recordings.

    Attributes
    ----------
    MSEED_path : pathlib Path object
        Location of raw mSEED files
    format : str
        File naming format of data archive
    signal :

    filtered_signal :

    stations : pandas Series object
        Series object containing station names
    st : obspy Stream object
        List like object containing available traces for given stations
    resample : bool, optional

    Methods
    -------
    path_structure(path_type="YEAR/JD/STATION")
        Set the file naming format of the data archive
    read_mseed(start_time, end_time, sampling_rate)
        Read in mSEED data between two times

    """

    def __init__(self, LUT, HOST_PATH):
        """
        MSEED object initialisation

        Parameters
        ----------
        LUT : array-like
            Lookup table object (only used for station names)
        HOST_PATH : str
            Location of raw mSEED files

        """

        self.MSEED_path = pathlib.Path(HOST_PATH)

        self.format = None
        self.signal = None
        self.filtered_signal = None

        self.resample = False

        lut = cmod.LUT()
        lut.load(LUT)
        self.stations = lut.station_data["Name"]
        del lut
        self.st = None

    def __str__(self):
        """
        Return short summary string of the mSEED object

        It will provide information about the archive location and structure,
        data sampling rate and time period over which the archive is being
        queried.

        """

        out = "QuakeMigrate mSEED object"
        out += "\n\tHost path\t:\t{}".format(self.MSEED_path)
        out += "\n\tPath structure\t:\t{}".format(self.format)
        out += "\n\tResampling\t:\t{}".format(self.resample)
        # out += "\n\tSampling rate\t:\t{}".format(self.sampling_rate)
        # out += "\n\tStart time\t:\t{}".format(str(self.start_time))
        # out += "\n\tEnd time\t:\t{}".format(str(self.end_time))
        out += "\n\tStations:"
        for station in self.stations:
            out += "\n\t\t{}".format(station)

        return out

    def path_structure(self, path_type="YEAR/JD/STATION"):
        """
        Define the format of the data archive

        Parameters
        ----------
        path_type : str, optional
            Sets path type for different archive formats

        """

        if path_type == "SeisComp3":
            self.format = "{year}/*/{station}/*/*.{station}..*.D.{year}.{jday}"
        elif path_type == "YEAR/JD/*_STATION":
            self.format = "{year}/{jday}/*_{station}_*"
        elif path_type == "YEAR/JD/STATION":
            self.format = "{year}/{jday}/{station}*"
        elif path_type == "STATION.YEAR.JULIANDAY":
            self.format = "*{station}.*.{year}.{jday}"
        elif path_type == "/STATION/STATION.YearMonthDay":
            self.format = "{station}/{station}.{year}{month:02d}{day:02d}"
        elif path_type == "YEAR_JD/STATION":
            self.format = "{year}_{jday}/{station}_*"

    def read_mseed(self, start_time, end_time, sampling_rate):
        """
        Reading the required mSEED files for all stations between two times
        and return station availability of the seperate stations during this
        period

        Parameters
        ----------
        start_time : UTCDateTime object
            Start datetime to read mSEED
        end_time : UTCDateTime object
            End datetime to read mSEED
        sampling_rate : int
            Sampling rate in hertz

        """

        self.sampling_rate = sampling_rate
        self.start_time = start_time
        self.end_time = end_time

        samples = int(round((end_time - start_time) * sampling_rate + 1))
        files = self._load_from_path(start_time, end_time)

        st = obspy.Stream()
        try:
            first = next(files)
            files = chain([first], files)
            for file in files:
                file = str(file)
                try:
                    st += obspy.read(file, starttime=start_time,
                                     endtime=end_time)
                except TypeError:
                    msg = "Station file not mSEED - {}"
                    print(msg.format(file))
                    continue

            # Remove all stations with data gaps greater than
            # 10.0 milliseconds
            st.merge(method=-1)
            gaps = st.get_gaps()
            if gaps:
                stations = np.unique(np.array(gaps)[:, 1]).tolist()
                for station in stations:
                    traces = st.select(station=station)
                    for trace in traces:
                        st.remove(trace)

            # Combining the mseed and determining station availability
            st.detrend("linear")
            st.detrend("demean")
            st = self._downsample(st, sampling_rate)

            signal, availability = self._station_availability(st, samples)
        except StopIteration:
            print("No data exist for this time period - creating blank")
            availability = np.zeros((len(self.stations), 1))
            signal = np.zeros((3, len(self.stations), int(samples)))

        self.st = st
        self.signal = signal
        self.filtered_signal = np.empty((self.signal.shape))
        self.filtered_signal[:] = np.nan
        self.availability = availability

    def _station_availability(self, stream, samples):
        """
        Determine whether data exist between two times for a given receiver

        Parameters
        ----------
        stream : obspy Stream object
            Stream containing 3-component trace data for stations
        samples : int
            Number of samples expected in the signal

        Returns
        -------
        signal : array-like

        availability : array-like
            Array containing 0s (no data) or 1s (data)

        """

        availability = np.zeros((len(self.stations), 1))
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

            else:
                # print("Trace not continuously active during this period.")
                continue

        return signal, availability

    def _load_from_path(self, start_time, end_time):
        """
        Retrieves available files between two times

        Parameters
        ----------
        start_time : UTCDateTime object
            Start datetime to read mSEED
        end_time : UTCDateTime object
            End datetime to read mSEED

        Returns
        -------
        files : generator
            Iterator object of available mSEED files

        """

        if self.format is None:
            print("Specify the path structure using MSEED.path_structure")
            return

        dy = 0
        files = []
        start_day = obspy.UTCDateTime(
            "{}-{}T00:00:00.0".format(start_time.year,
                                      str(start_time.julday).zfill(3)))
        while start_day + (dy * 86400) <= end_time:
            now = start_time + (dy * 86400)
            for stat in self.stations.tolist():
                file_format = self.format.format(
                    year=now.year,
                    month=now.month,
                    jday=str(now.julday).zfill(3),
                    station=stat)
                files = chain(files, self.MSEED_path.glob(file_format))

            dy += 1

        return files

    def _downsample(self, stream, sr):
        """
        Downsample the MSEED to the designated sampling rate

        Parameters
        ----------
        stream : obspy Stream object
            Contains list of Trace objects to be downsampled
        sr : int
            Designated sample rate

        """

        for trace in stream:
            if sr != trace.stats.sampling_rate:
                trace.filter(
                    "lowpass",
                    freq=float(sr) / 2.000001,
                    corners=2,
                    zerophase=True)
                if (trace.stats.sampling_rate % sr) == 0:
                    trace.decimate(
                        factor=int(trace.stats.sampling_rate / sr),
                        strict_length=False,
                        no_filter=True)
                elif self.resample:
                    # trace.resample(
                    #     sr,
                    #     strict_length=False,
                    #     no_filter=True)
                    trace.interpolate(sr)

                else:
                    msg = "Mismatched sampling rates - cannot decimate data.\n"
                    msg += "To resample data, set .resample = True"
                    print(msg)

        return stream

    @property
    def sample_size(self):
        """Get the size of a sample (units: s)"""

        return 1 / self.sampling_rate
