# -*- coding: utf-8 -*-
"""
Module to handle input/output of TriggeredEvents.csv files.

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
from datetime import time

from obspy import UTCDateTime
import pandas as pd

import quakemigrate.util as util


OUTPUT_COLS = [
    "EventID",
    "CoaTime",
    "TRIG_COA",
    "COA_X",
    "COA_Y",
    "COA_Z",
    "COA",
    "COA_NORM",
]


def read_triggered_events(run, **kwargs):
    """
    Read triggered events from .csv file.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.core.Run` object
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
        Triggered events information. Columns: ["EventID", "CoaTime", "TRIG_COA",
        "COA_X", "COA_Y", "COA_Z", "COA", "COA_NORM"].

    """

    starttime = kwargs.get("starttime", None)
    endtime = kwargs.get("endtime", None)
    trigger_file = kwargs.get("trigger_file", None)

    fpath = run.path / "trigger" / run.subname / "events"

    if trigger_file is not None:
        events = pd.read_csv(trigger_file)
    else:
        trigger_files = []
        readstart = UTCDateTime(starttime.date)
        while readstart <= endtime:
            fstem = f"{run.name}_{readstart.year}_{readstart.julday:03d}"
            file = (fpath / f"{fstem}_TriggeredEvents").with_suffix(".csv")
            if file.is_file():
                trigger_files.append(file)
            else:
                logging.info(f"\n\t    Cannot find file: {fstem}")
            readstart += 86400
        if len(trigger_files) == 0:
            raise util.NoTriggerFilesFound
        events = pd.concat((pd.read_csv(f) for f in trigger_files), ignore_index=True)

    events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)

    if starttime is not None and endtime is not None:
        # Check if the batch extends to midnight; if so, use "less than" condition to
        # ensure consistent treatment of multi-day runs (midnight = next day, so not
        # included). We do not have access to the detect scan rate here any longer, but
        # using "less than" is sufficient, and avoids hard-coding a (minimum) scan_rate
        # sampling interval.
        if endtime.time == time(0, 0):
            events = events[
                (events["CoaTime"] >= starttime) & (events["CoaTime"] < endtime)
            ]
        else:
            events = events[
                (events["CoaTime"] >= starttime) & (events["CoaTime"] <= endtime)
            ]

    if len(events) == 0:
        logging.info(
            "\n\t    No triggered events found! Check your trigger output files.\n"
        )

    return events.reset_index()


@util.timeit("info")
def write_triggered_events(run, events, starttime, write_event_time_windows):
    """
    Write triggered events to a .csv file.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    events : `pandas.DataFrame` object
        Triggered events information. Columns: ["EventID", "CoaTime", "TRIG_COA",
        "COA_X", "COA_Y", "COA_Z", "COA", "COA_NORM"].
    starttime : `obspy.UTCDateTime` object
        Timestamp from which events have been triggered.
    write_event_time_windows: bool
        If true, retain the MinTime and MaxTime columns which denote the trigger window
        for each candidate event.

    """

    fpath = run.path / "trigger" / run.subname / "events"
    fpath.mkdir(exist_ok=True, parents=True)

    output_cols = OUTPUT_COLS
    if write_event_time_windows:
        output_cols.extend(["MinTime", "MaxTime"])

    # Work on a copy
    events = events.copy()
    events = events.loc[:, output_cols]

    fstem = f"{run.name}_{starttime.year}_{starttime.julday:03d}"
    file = (fpath / f"{fstem}_TriggeredEvents").with_suffix(".csv")
    events.to_csv(file, index=False)
