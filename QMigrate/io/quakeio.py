# -*- coding: utf-8 -*-
"""
Module to produce gridded traveltime velocity models

"""

import os
import pathlib
from datetime import datetime

import obspy
from obspy import Stream, Trace, UTCDateTime
import pandas as pd
import numpy as np
import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")


class QuakeIO:
    """
    Input / output control class

    Provides the basic methods for input / output of seismic scan files

    Attributes
    ----------
    path : pathlib Path object
        Location of input/output files
    file_sample_rate : float
        Sample rate (units: ms)

    Methods
    -------
    read_scan()
        Parse information from an existing .scn file
    write_scan(daten, dsnr, dloc)
        Create a new .scn file
    del_scan()
        Delete an existing .scn file

    """

    def __init__(self, path="", name=None):
        """
        Class initialisation method

        Parameters
        ----------
        path : str, optional

        name: str, optional

        """

        path = pathlib.Path(path)
        if name is None:
            name = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
        print("Path = {}, Name = {}".format(str(path), name))
        self.path = path
        self.name = name

        self.file_sample_rate = None

    def del_scan(self):
        """
        Delete an existing .scn file

        """

        fname = (self.path / self.name).with_suffix(".scn")
        if fname.exists():
            print("Filename {} already exists - deleting.".format(str(fname)))
            fname.unlink()

    def read_coal4D(self, fname):
        """
        Reads a binary file

        Parameters
        ----------
        fname : str or pathlib.Path object
            Location of file to be read

        """

        map_ = np.load(fname)
        return map_

    def write_coal4D(self, map_, event, start_time, end_time):
        """
        Outputs a binary file

        Parameters
        ----------
        map_ :

        event :

        start_time :

        end_time :

        """

        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)
        fname = self.path / self.name / "{}_{}_{}".format(event,
                                                          start_time,
                                                          end_time)
        fname = fname.with_suffix(".coal4D")

        np.save(str(fname), map_)

    def read_decscan(self):
        """

        """

        fname = (self.path / self.name).with_suffix(".scnmseed")
        coa = obspy.read(str(fname))
        coa_stats = coa.select(station="COA")[0].stats

        data = pd.DataFrame()

        td = 1 / coa_stats.sampling_rate
        tmp = np.arange(coa_stats.starttime,
                        coa_stats.endtime + td,
                        td)
        data["DT"] = [x.datetime for x in tmp]
        data["COA"] = coa.select(station="COA")[0].data / 1e5
        data["COA_N"] = coa.select(station="COA_N")[0].data / 1e5
        data["X"] = coa.select(station="X")[0].data / 1e6
        data["Y"] = coa.select(station="Y")[0].data / 1e6
        data["Z"] = coa.select(station="Z")[0].data / 1e3

        data["DT"] = data["DT"].apply(UTCDateTime)

        return data

    def write_decscan(self, original_dataset, daten, dsnr, dsnr_norm, dloc, sampling_rate):
        """
        Create a new .scnmseed file

        Parameters
        ----------
        sampling_rate : int
            Sampling rate in hertz

        """

        fname = (self.path / self.name).with_suffix(".scnmseed")

        dsnr[dsnr > 21474.] = 21474.
        dsnr_norm[dsnr_norm > 21474.] = 21474.

        npts = len(dsnr)
        starttime = UTCDateTime(daten[0])
        meta = {"network": "NW",
                "npts": npts,
                "sampling_rate": sampling_rate,
                "starttime": starttime}

        st = Stream(Trace(data=(dsnr * 1e5).astype(np.int32),
                          header={**{"station": "COA"}, **meta}))
        st += Stream(Trace(data=(dsnr_norm * 1e5).astype(np.int32),
                           header={**{"station": "COA_N"}, **meta}))
        st += Stream(Trace(data=(dloc[:, 0] * 1e6).astype(np.int32),
                           header={**{"station": "X"}, **meta}))
        st += Stream(Trace(data=(dloc[:, 1] * 1e6).astype(np.int32),
                           header={**{"station": "Y"}, **meta}))
        st += Stream(Trace(data=(dloc[:, 2] * 1e3).astype(np.int32),
                           header={**{"station": "Z"}, **meta}))

        if original_dataset is not None:
            original_dataset = original_dataset + st
        else:
            original_dataset = st

        original_dataset.write(str(fname), format="MSEED", encoding=11)

        return original_dataset

    def write_log(self, message):
        """
        Method that tracks the progress of a scanning run through time

        Parameters
        ----------
        message : str
            Information to be saved to the log file

        """

        fname = (self.path / self.name).with_suffix(".log")
        with fname.open(mode="a") as f:
            f.write(message + "\n")

    def cut_mseed(self, data, event_name):
        """
        Output a mSEED file

        Parameters
        ----------
        data :

        event_name : str
            Event ID

        """

        fname = self.path / "{}_{}".format(self.name, event_name)
        fname = str(fname.with_suffix(".mseed"))
        st = data.st
        st.write(str(fname), format="MSEED")

    def write_stations_file(self, stations, event_name):
        """
        Create a new .stn file

        Parameters
        ----------
        stations : pandas DataFrame object

        event_name : str


        """

        fname = self.path / "{}_{}".format(self.name, event_name)
        fname = str(fname.with_suffix(".stn"))
        stations.to_csv(fname, index=False)

    def write_event(self, event, event_name):
        """
        Create a new .event file

        Parameters
        ----------
        events : pandas DataFrame object

        event_name : str


        """

        fname = self.path / "{}_{}".format(self.name, event_name)
        fname = str(fname.with_suffix(".event"))
        event.to_csv(fname, index=False)

    def read_triggered_events(self, start_time, end_time):
        """

        Parameters
        ----------
        start_time : UTCDateTime object

        end_time : UTCDateTime object

        """
        fname = self.path / "{}_TriggeredEvents".format(self.name)
        fname = str(fname.with_suffix(".csv"))
        events = pd.read_csv(fname)

        # Trim the events between the start and end times
        events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)
        events = events[(events["CoaTime"] >= start_time) &
                        (events["CoaTime"] <= end_time)]

        events["MinTime"] = events["MinTime"].apply(UTCDateTime)
        events["MaxTime"] = events["MaxTime"].apply(UTCDateTime)

        return events

    def write_triggered_events(self, events):
        """
        Create a new triggered events csv

        Parameters
        ----------
        events : pandas DataFrame object
            Contains information on triggered events

        """

        fname = self.path / "{}_TriggeredEvents".format(self.name)
        fname = str(fname.with_suffix(".csv"))
        events.to_csv(fname, index=False)