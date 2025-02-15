"""
Module to handle input/output of TriggeredEvents.csv files.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from datetime import time

import pandas as pd
from obspy import UTCDateTime

import quakemigrate
import quakemigrate.util as util


def read_triggered_events(
    run: quakemigrate.io.core.Run,
    starttime: UTCDateTime | None = None,
    endtime: UTCDateTime | None = None,
    trigger_file: str | None = None,
) -> pd.DataFrame:
    """
    Read triggered events from .csv file.

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    starttime:
        Timestamp from which to include events in the locate scan.
    endtime:
        Timestamp up to which to include events in the locate scan.
    trigger_file:
        File containing triggered events to be located.

    Returns
    -------
     :
        Triggered events information. Columns: ["EventID", "CoaTime", "TRIG_COA",
        "COA_X", "COA_Y", "COA_Z", "COA", "COA_NORM"].

    """

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
def write_triggered_events(
    run: quakemigrate.io.core.Run, events: pd.DataFrame, starttime: UTCDateTime
) -> None:
    """
    Write triggered events to a .csv file.

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    events:
        Triggered events information. Columns: ["EventID", "CoaTime", "TRIG_COA",
        "COA_X", "COA_Y", "COA_Z", "COA", "COA_NORM"].
    starttime:
        Timestamp from which events have been triggered.

    """

    fpath = run.path / "trigger" / run.subname / "events"
    fpath.mkdir(exist_ok=True, parents=True)

    # Work on a copy
    events = events.copy()
    events = events.loc[
        :,
        [
            "EventID",
            "CoaTime",
            "TRIG_COA",
            "COA_X",
            "COA_Y",
            "COA_Z",
            "COA",
            "COA_NORM",
        ],
    ]

    fstem = f"{run.name}_{starttime.year}_{starttime.julday:03d}"
    file = (fpath / f"{fstem}_TriggeredEvents").with_suffix(".csv")
    events.to_csv(file, index=False)
