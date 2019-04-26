# -*- coding: utf-8 -*-
"""
Module for processing mSEED files.

Provides information on station in a given archive.

"""

import obspy
import pathlib
import QMigrate.core.model as cmod
import numpy as np


class MSEED():
    """
    MSEED object

    Contains

    Attributes
    ----------
    MSEED_path : str
        String describing location of mSEED files
    format : str
        File naming format of data archive

    Methods
    -------
    path_structure(path_type="YEAR/JD/STATION")
        Set the file naming format of the data archive

    """

    def __init__(self, LUT, HOST_PATH='/PATH/MSEED'):
        """
        MSEED object initialisation

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

    def path_structure(self, path_type='YEAR/JD/STATION'):
        """
        Define the format of the data archive

        Parameters
        ----------
        path_type : str

        """

        if path_type == "SeisComp3":
            self.format = "{year}/*/{station}/*/*.{station}..*.D.{year}.{jday}"
        elif path_type == "YEAR/JD/STATION":
            self.format = "{year}/{jday}/*_{station}_*"
        elif path_type == 'STATION.YEAR.JULIANDAY':
            self.format = "*{station}.*.{year}.{jday}"
        elif path_type == "/STATION/STATION.YearMonthDay":
            self.format = "{station}/{station}.{year}{month:02d}{day:02d}"

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
        samples = int((end_time - start_time) * sampling_rate + 1)
        files = self._load_from_path()

        st = obspy.Stream()

        if files:
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
                st._cleanup()
                gaps = st.get_gaps()
                if len(gaps) > 0:
                    stats_with_gaps = np.unique(np.array(gaps)[:, 1]).tolist()
                    for station in stats_with_gaps:
                        traces = st.select(station=station)
                        for trace in traces:
                            st.remove(trace)

                # Combining the mseed and determining station availability
                st.detrend("linear")
                st.detrend("demean")
                st = self._downsample(st, sampling_rate)

                signal, availability = self._station_availability(st, samples)
        else:
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

        samples : int
            Number of samples expected in the signal

        Returns
        -------
        signal : array-like

        availability : array-like


        """

        availability = np.zeros((len(self.stations), 1))
        signal = np.zeros((3, len(self.stations), int(samples)))

        for i, station in enumerate(self.stations):
            tmp_st = stream.select(station=station)
            if len(tmp_st) == 3:
                if (tmp_st[0].stats.npts == samples and
                        tmp_st[1].stats.npts == samples and
                        tmp_st[2].stats.npts == samples):

                    # Defining the station as avaliable
                    availability[i] = 1

                    for tr in tmp_st:
                        channel = tr.stats.channel[-1]
                        if channel == 'E' or channel == '2':
                            signal[1, i, :] = tr.data

                        if channel == 'N' or channel == '1':
                            signal[0, i, :] = tr.data

                        if channel == 'Z':
                            signal[2, i, :] = tr.data

            else:
                # print("Trace not continuously active during this period.")
                continue

        return signal, availability

    def _load_from_path(self):
        """
        Retrieves a list of files between two times

        Returns
        -------
        FILES : list
            List of available mSEED files

        """

        if self.format is None:
            print('Please specfiy the path_structure - DATA.path_structure')
            return

        dy = 0
        FILES = []
        while (self.start_time + (dy * 86400)).julday <= self.end_time.julday:
            now = self.start_time + (dy * 86400)
            for stat in self.stations.tolist():
                file_format = self.format.format(
                    year=now.year,
                    month=now.month,
                    jday=str(now.julday).zfill(3),
                    station=stat)
                files = list(self.MSEED_path.glob(file_format))
                FILES.extend(files)

                dy += 1

        return FILES

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
                    trace.resample(
                        sr,
                        strict_length=False,
                        no_filter=True)
                else:
                    msg = "Mismatched sampling rates - cannot decimate data.\n"
                    msg += "To resample data, set .resample = True"
                    print(msg)

        return stream

    @property
    def sample_size(self):
        """Get the size of a sample (units: s)"""

        return 1 / self.sampling_rate
