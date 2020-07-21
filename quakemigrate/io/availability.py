# -*- coding: utf-8 -*-
"""
Module to handle input/output of StationAvailability.csv files.

:copyright:
    2020, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

from obspy import UTCDateTime
import pandas as pd

import quakemigrate.util as util


def read_availability(run, starttime, endtime):
    """
    Read in station availability data to a `pandas.DataFrame` from csv files
    split by Julian day.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    starttime : `obspy.UTCDateTime` object
        Timestamp from which to read the station availability.
    endtime : `obspy.UTCDateTime` object
        Timestamp up to which to read the station availability.

    Returns
    -------
    availability : `pandas.DataFrame` object
        Details the availability of each station for each timestep of detect.

    """

    fpath = run.path / "detect" / "availability"

    startday = UTCDateTime(starttime.date)

    dy = 0
    availability = None
    # Loop through days trying to read .StationAvailability files
    logging.debug("\t    Reading in .StationAvailability...")
    while startday + (dy * 86400) <= endtime:
        now = starttime + (dy * 86400)
        fstem = f"{now.year}_{now.julday:03d}_StationAvailability"
        file = (fpath / fstem).with_suffix(".csv")
        try:
            if availability is None:
                availability = pd.read_csv(file, index_col=0)
            else:
                tmp = pd.read_csv(file, index_col=0)
                availability = pd.concat([availability, tmp])
        except FileNotFoundError:
            logging.info("\tNo .StationAvailability file found for "
                         f"{now.year} - {now.julday:03d}")
        dy += 1

    if availability is None:
        raise util.NoStationAvailabilityDataException

    starttime, endtime = availability.index[0], availability.index[-1]
    logging.debug(f"\t\t...from {starttime} - {endtime}")


    return availability


def write_availability(run, availability):
    """
    Write out csv files (split by Julian day) containing station availability
    data.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    availability : `pandas.DataFrame` object
        Details the availability of each station for each timestep of detect.

    """

    fpath = run.path / "detect" / "availability"
    fpath.mkdir(exist_ok=True, parents=True)

    availability.index = times = pd.to_datetime(availability.index)
    datelist = set([time.date() for time in times])

    for date in datelist:
        to_write = availability[availability.index.date == date]
        to_write.index = [UTCDateTime(idx) for idx in to_write.index]
        date = UTCDateTime(date)

        fstem = f"{date.year}_{date.julday:03d}_StationAvailability"
        file = (fpath / fstem).with_suffix(".csv")

        to_write.to_csv(file)
