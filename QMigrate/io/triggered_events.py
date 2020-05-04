# -*- coding: utf-8 -*-
"""
Module to handle input/output of TriggeredEvents.csv files.

"""

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
        file = trigger_file
    else:
        fstem = f"{run.name}_TriggeredEvents"
        file = (fpath / fstem).with_suffix(".csv")

    events = pd.read_csv(file)

    events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)
    events["MinTime"] = events["MinTime"].apply(UTCDateTime)
    events["MaxTime"] = events["MaxTime"].apply(UTCDateTime)

    if starttime is not None and endtime is not None:
        events = events[(events["CoaTime"] >= starttime) &
                        (events["CoaTime"] <= endtime)]

    return events.reset_index()


def write_triggered_events(run, events, start_time, end_time):
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

    """

    fpath = run.path / "trigger" / run.subname / "events"
    fpath.mkdir(exist_ok=True, parents=True)

    fstem = f"{run.name}_TriggeredEvents"
    file = (fpath / fstem).with_suffix(".csv")
    events.to_csv(file, index=False)
