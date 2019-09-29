# -*- coding: utf-8 -*-
"""
Module to handle input/output for QuakeMigrate.

"""

from datetime import datetime
import pathlib

import numpy as np
from obspy import Stream, UTCDateTime, read
import pandas as pd

import QMigrate.util as util


def stations(station_file, delimiter=","):
    """
    Reads station information from file.

    Parameters
    ----------
    station_file : str
        Path to station file. File format:
        Header line is REQUIRED: Latitude,Longitude,Elevation,Name (any order).
        Elevation in METRES.

    delimiter : char, optional
        Station file delimiter, defaults to ","

    Returns
    -------
    stn_data : Pandas DataFrame object
        Columns: "Latitude", "Longitude", "Elevation", "Name"

    """

    stn_data = pd.read_csv(station_file, delimiter=delimiter)

    if ("Latitude" or "Longitude" or "Elevation" or "Name") \
       not in stn_data.columns:
        raise util.StationFileHeaderException

    return stn_data


class QuakeIO:
    """
    Input / output control class

    Provides the basic methods for input / output of QuakeMigrate files.

    Attributes
    ----------
    path : pathlib Path object
        Location of input/output files

    name : str, optional
        Run name

    Methods
    -------
    read_coal4D()
        Read binary numpy 4D coalescence map file

    write_coal4D()
        Write 4D coalescence map to binary numpy file

    read_coastream()
        Read an existing .scanmseed file

    write_coastream()
        Write a new .scanmseed file

    write_log()
        Track progress of run and write to log file

    write_cut_waveforms(format="MSEED")
        Write raw cut waveform data (defaults to mSEED format)

    write_picks()
        Write phase pick data

    write_event()
        Write located event data

    read_triggered_events()
        Read triggered events from file

    write_triggered_events()
        Write triggered events to file

    """

    def __init__(self, path, name=None, log=False):
        """
        Class initialisation method.

        Parameters
        ----------
        path : str
            Path to input / output directory

        name: str, optional
            Name of run

        """

        path = pathlib.Path(path)
        if name is None:
            name = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
        self.path = path
        self.name = name
        self.run = path / name

        self.log_ = log

        # Make run output directory
        util.make_directories(self.run)

    def read_coal4D(self, fname):
        """
        Reads a binary numpy file containing 4-D coalescence grid output by
        _compute() .

        Parameters
        ----------
        fname : str or pathlib.Path object
            Location of file to be read

        Returns
        -------
        map_4d : array-like
            4-D coalescence grid

        """

        map_4d = np.load(fname)

        return map_4d

    def write_coal4D(self, map_4d, event_name, start_time, end_time):
        """
        Writes 4-D coalescence grid to a binary numpy file.

        Parameters
        ----------
        map_4d : array-like
            4-D coalescence grid output by _compute()

        event_name : str
            event_id for file naming

        start_time : UTCDateTime
            start time of 4-D coalescence map

        end_time : UTCDateTime
            end time of 4-D coalescence map

        """

        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        subdir = "4d_coal_grids"
        util.make_directories(self.run, subdir=subdir)
        filestr = "{}_{}_{}_{}".format(self.name, event_name, start_time,
                                       end_time)
        fname = self.run / subdir / filestr
        fname = fname.with_suffix(".coal4D")

        np.save(str(fname), map_4d)

    def read_coastream(self, start_time, end_time):
        """
        Read coastream data from .scanmseed files between two time stamps.
        Files are labelled by year and julian day.

        Parameters
        ----------
        start_time : UTCDateTime object
            start time to read coastream from

        end_time : UTCDateTime obbject
            end time to read coastream to

        Returns
        -------
        data : pandas DataFrame
            Data output by detect() -- decimated scan
            Columns: ["COA", "COA_N", "X", "Y", "Z"] - X & Y as lon/lat; z in m

        coa_stats : obspy Trace stats object
            obspy Trace stats of raw coalescence trace ("COA").
            Contains keys: network, station, channel, starttime, endtime,
                           sampling_rate, delta, npts, calib, _format, mseed

        """

        start_day = UTCDateTime(start_time.date)

        dy = 0
        files = []
        coa = Stream()
        # Loop through days trying to read coastream files
        while start_day + (dy * 86400) <= end_time:
            now = start_time + (dy * 86400)
            filestr = "{}_{}_{}".format(self.name, str(now.year),
                                        str(now.julday).zfill(3))
            fname = (self.run / filestr).with_suffix(".scanmseed")
            try:
                coa += read(str(fname), starttime=start_time,
                            endtime=end_time, format="MSEED")
            except FileNotFoundError:
                msg = "\tNo .scanmseed file found for day {}-{}!\n"
                msg = msg.format(str(now.year), str(now.julday).zfill(3))
                self.log(msg, self.log_)

            dy += 1

        if not bool(coa):
            raise util.NoScanMseedDataException

        coa.merge(method=-1)
        coa_stats = coa.select(station="COA")[0].stats

        msg = "\t\tSuccessfully read .scanmseed data from {} - {}\n"
        msg = msg.format(str(coa_stats.starttime), str(coa_stats.endtime))
        self.log(msg, self.log_)

        data = pd.DataFrame()

        td = 1 / coa_stats.sampling_rate
        data["DT"] = np.arange(coa_stats.starttime,
                               coa_stats.endtime + td,
                               td)

        # assign to DataFrame column and divide by factor applied in
        # write_coastream()
        data["COA"] = coa.select(station="COA")[0].data / 1e5
        data["COA_N"] = coa.select(station="COA_N")[0].data / 1e5
        data["X"] = coa.select(station="X")[0].data / 1e6
        data["Y"] = coa.select(station="Y")[0].data / 1e6
        data["Z"] = coa.select(station="Z")[0].data / 1e3

        return data, coa_stats

    def write_coastream(self, st, write_start=None, write_end=None):
        """
        Write a new .scanmseed file from an obspy Stream object containing the
        data output from detect(). Note: values have been multiplied by a
        power of ten, rounded and converted to an int32 array so the data can
        be saved as mSEED with STEIM2 compression. This multiplication factor
        is removed when the data is read back in with read_coastream().

        Files are labelled by year and julian day, and split by julian day
        (this behaviour is determined in signal/scan.py).

        Parameters
        ----------
        st : obspy Stream object
            Output of detect() stored in obspy Stream object with
            channels: ["COA", "COA_N", "X", "Y", "Z"]

        write_start : UTCDateTime object, optional
            Time from which to write the coastream stream to a file

        write_end : UTCDateTime object, optional
            Time upto which to write the coastream stream to a file

        """

        if write_start or write_end:
            st = st.slice(starttime=write_start, endtime=write_end)

        file_str = "{}_{}_{}".format(self.name,
                                     str(st[0].stats.starttime.year),
                                     str(st[0].stats.starttime.julday).zfill(3))
        fname = (self.run / file_str).with_suffix(".scanmseed")

        st.write(str(fname), format="MSEED", encoding="STEIM2")

    def log(self, message, log):
        """
        Write a log file to track the progress of a scanning run through time.

        Parameters
        ----------
        message : str
            Information to be saved to the log file

        """

        print(message)

        if log:
            subdir = "logs"
            util.make_directories(self.run, subdir=subdir)
            fname = (self.run / subdir / self.name).with_suffix(".log")
            with fname.open(mode="a") as f:
                f.write(message + "\n")

    def write_cut_waveforms(self, data, event, event_name, data_format="MSEED",
                            pre_cut=None, post_cut=None):
        """
        Output raw cut waveform data as a waveform file -- defaults to mSEED.

        Parameters
        ----------
        data : Archive object
            Contains read_waveform_data() method and stores read data in raw
            and processed state

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

        event_name : str
            event_uid for file naming

        format : str, optional
            File format to write waveform data to. Options are all file formats
            supported by obspy, including: "MSEED" (default), "SAC", "SEGY",
            "GSE2"

        pre_cut : float, optional
            Specify how long before the event origin time to cut the waveform
            data from

        post_cut : float, optional
            Specify how long after the event origin time to cut the waveform
            data to

        """

        st = data.raw_waveforms

        otime = UTCDateTime(event["DT"])
        if pre_cut:
            for tr in st.traces:
                tr.trim(starttime=otime - pre_cut)
        if post_cut:
            for tr in st.traces:
                tr.trim(endtime=otime + post_cut)

        subdir = "cut_waveforms"
        util.make_directories(self.run, subdir=subdir)
        fname = self.run / subdir / "{}".format(event_name)

        if data_format == "MSEED":
            suffix = ".m"
        elif data_format == "SAC":
            suffix = ".sac"
        elif data_format == "SEGY":
            suffix = ".segy"
        elif data_format == "GSE2":
            suffix = ".gse2"
        else:
            suffix = ".waveforms"

        fname = str(fname.with_suffix(suffix))
        st.write(str(fname), format=data_format)  # , encoding="STEIM1")

    def write_picks(self, phase_picks, event_name):
        """
        Write phase picks to a new .picks file

        Parameters
        ----------
        phase_picks : pandas DataFrame object

        event_name : str
            event ID for file naming

        """

        subdir = "picks"
        util.make_directories(self.run, subdir=subdir)
        fname = self.run / subdir / "{}".format(event_name)
        fname = str(fname.with_suffix(".picks"))
        phase_picks.to_csv(fname, index=False)

    def write_amplitudes(self, amplitudes, event_name):
        """
        Write phase picks to a new .picks file

        Parameters
        ----------
        amplitudes : pandas DataFrame object

        event_name : str
            event ID for file naming

        """

        subdir = "amplitudes"
        util.make_directories(self.run, subdir=subdir)
        fname = self.run / subdir / "{}".format(event_name)
        fname = str(fname.with_suffix(".amps"))
        amplitudes.to_csv(fname, index=False)

    def write_event(self, event, event_name):
        """
        Create a new .event file

        Parameters
        ----------
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

        event_name : str
            event_uid for file naming

        """

        subdir = "events"
        util.make_directories(self.run, subdir=subdir)
        fname = self.run / subdir / "{}".format(event_name)
        fname = str(fname.with_suffix(".event"))
        event.to_csv(fname, index=False)

    def read_triggered_events_time(self, start_time, end_time):
        """
        Read triggered events output by trigger() from csv file.

        Parameters
        ----------
        start_time : UTCDateTime object

        end_time : UTCDateTime object

        Returns
        -------
        events : pandas DataFrame
            Triggered events output from _trigger_scn().
            Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                      "COA_Z", "MinTime", "MaxTime"]

        """
        fname = self.run / "{}_TriggeredEvents".format(self.name)
        fname = str(fname.with_suffix(".csv"))
        events = pd.read_csv(fname)

        # Trim the events between the start and end times
        events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)
        events = events[(events["CoaTime"] >= start_time) &
                        (events["CoaTime"] <= end_time)]

        events["MinTime"] = events["MinTime"].apply(UTCDateTime)
        events["MaxTime"] = events["MaxTime"].apply(UTCDateTime)

        return events

    def read_triggered_events(self, fname):
        """
        Read triggered events output by trigger() from csv file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        events : pandas DataFrame
            Triggered events output from _trigger_scn().
            Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                      "COA_Z", "MinTime", "MaxTime"]

        """
        events = pd.read_csv(fname)

        events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)
        events["MinTime"] = events["MinTime"].apply(UTCDateTime)
        events["MaxTime"] = events["MaxTime"].apply(UTCDateTime)

        return events

    def write_triggered_events(self, events, start_time, end_time):
        """
        Write triggered events output by trigger() to a csv file.

        Parameters
        ----------
        events : pandas DataFrame
            Triggered events output from _trigger_scn().
            Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                      "COA_Z", "MinTime", "MaxTime"]

        """

        fname = self.run / "{}_{}-{}_TriggeredEvents".format(self.name,
                                                             start_time.julday,
                                                             end_time.julday)
        fname = str(fname.with_suffix(".csv"))
        events.to_csv(fname, index=False)

    def write_stn_availability(self, stn_ava_data):
        """
        Write out csv files (split by julian day) containing station
        availability data.

        Parameters
        ----------
        stn_ava_data : pandas DataFrame object
            Contains station availability information

        """

        stn_ava_data.index = times = pd.to_datetime(stn_ava_data.index)
        datelist = [time.date() for time in times]

        for date in datelist:
            to_write = stn_ava_data[stn_ava_data.index.date == date]
            to_write.index = [UTCDateTime(idx) for idx in to_write.index]
            date = UTCDateTime(date)
            fname = self.run / "{}_{}_{}_StationAvailability".format(self.name,
                                                                     date.year,
                                                                     str(date.julday).zfill(3))
            fname = str(fname.with_suffix(".csv"))
            to_write.to_csv(fname)

    def read_stn_availability(self, start_time, end_time):
        """
        Read in station availability data to a pandas DataFrame
        from csv files split by Julian day.

        Parameters
        ----------
        start_time : UTCDateTime object
            start time to read stn_ava_data from

        end_time : UTCDateTime obbject
            end time to read stn_ava_data to

        Returns
        -------
        stn_ava_data : pandas DataFrame object
            Contains station availability information

        """

        start_day = UTCDateTime(start_time.date)

        dy = 0
        stn_ava_data = None
        # Loop through days trying to read .StationAvailability files
        while start_day + (dy * 86400) <= end_time:
            now = start_time + (dy * 86400)
            filestr = "{}_{}_{}_StationAvailability".format(self.name,
                                                            str(now.year),
                                                            str(now.julday).zfill(3))
            fname = (self.run / filestr).with_suffix(".csv")
            try:
                if stn_ava_data is None:
                    stn_ava_data = pd.read_csv(fname, index_col=0)
                else:
                    tmp = pd.read_csv(fname, index_col=0)
                    stn_ava_data = pd.concat([stn_ava_data, tmp])
            except FileNotFoundError:
                msg = "\tNo .StationAvailability file found for day {}-{}!\n"
                msg = msg.format(str(now.year), str(now.julday).zfill(3))
                self.log(msg, self.log_)

            dy += 1

        if stn_ava_data is None:
            raise util.NoStationAvailabilityDataException

        msg = "\t\tSuccessfully read .StationAvailability data from {} - {}\n"
        msg = msg.format(stn_ava_data.index[0], stn_ava_data.index[-1])
        self.log(msg, self.log_)

        return stn_ava_data
