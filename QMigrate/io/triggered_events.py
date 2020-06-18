# -*- coding: utf-8 -*-
"""
Module to handle input/output of TriggeredEvents.csv files.

"""

import logging

from obspy import UTCDateTime
import pandas as pd


def read_triggered_events(run, **kwargs):
    """
    Read triggered events from .csv file.

    Parameters
    ----------
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    starttime : `obspy.UTCDateTime` object, optional
        Timestamp from which to include events in the locate scan.
    endtime : `obspy.UTCDateTime` object, optional
        Timestamp up to which to include events in the locate scan.
    trigger_file : str, optional
        File containing triggered events to be located.

    Returns
    -------
    events : `pandas.DataFrame` object
        Triggered events information.
        Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                  "MinTime", "MaxTime", "COA", "COA_NORM", "EventID"].

    """

    starttime = kwargs.get("starttime", None)
    endtime = kwargs.get("endtime", None)
    trigger_file = kwargs.get("trigger_file", None)

    fpath = run.path / "trigger" / run.subname / "events"

    if trigger_file is not None:
        events = pd.read_csv(trigger_file)
    else:
        trigger_files = []
        readstart = starttime
        while readstart <= endtime:
            fstem = f"{run.name}_{readstart.year}_{readstart.julday:03d}"
            file = (fpath / f"{fstem}_TriggeredEvents").with_suffix(".csv")
            if file.is_file():
                trigger_files.append(file)
            else:
                logging.info(f"\n\t    Cannot find file: {fstem}")
            readstart += 86400
        events = pd.concat((pd.read_csv(f) for f in trigger_files),
                           ignore_index=True)

    events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)
    events["MinTime"] = events["MinTime"].apply(UTCDateTime)
    events["MaxTime"] = events["MaxTime"].apply(UTCDateTime)

    if starttime is not None and endtime is not None:
        events = events[(events["CoaTime"] >= starttime) &
                        (events["CoaTime"] <= endtime)]

    return events.reset_index()


def write_triggered_events(run, events, starttime):
    """
    Write triggered events to a .csv file.

    Parameters
    ----------
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    events : `pandas.DataFrame` object
        Triggered events information.
        Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                  "MinTime", "MaxTime", "COA", "COA_NORM", "EventID"].
    starttime : `obspy.UTCDateTime` object
        Timestamp from which events have been triggered.

    """

    fpath = run.path / "trigger" / run.subname / "events"
    fpath.mkdir(exist_ok=True, parents=True)

    fstem = f"{run.name}_{starttime.year}_{starttime.julday:03d}"
    file = (fpath / f"{fstem}_TriggeredEvents").with_suffix(".csv")
    events.to_csv(file, index=False)
