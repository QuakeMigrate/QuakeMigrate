"""
Module to handle input/output of StationAvailability.csv files.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
from obspy import UTCDateTime

from quakemigrate.exceptions import NoStationAvailabilityData


if TYPE_CHECKING:
    from quakemigrate.io.core import Run


def read_availability(
    run: Run, starttime: UTCDateTime, endtime: UTCDateTime
) -> pd.DataFrame:
    """
    Read in station availability data to a `pandas.DataFrame` from csv files split by
    Julian day.

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    starttime:
        Timestamp from which to read the station availability.
    endtime:
        Timestamp up to which to read the station availability.

    Returns
    -------
    availability:
        Details the availability of each station for each timestep of detect.

    Raises
    ------
    NoStationAvailabilityData
        If no _StationAvailability.csv files are found.

    """

    fpath = run.path / "detect" / "availability"

    availability = None
    # Loop through days trying to read .StationAvailability files
    logging.debug("\t    Reading in .StationAvailability...")
    readstart = UTCDateTime(starttime.date)
    while readstart <= endtime:
        fstem = f"{readstart.year}_{readstart.julday:03d}"
        file = (fpath / f"{fstem}_StationAvailability").with_suffix(".csv")
        try:
            if availability is None:
                availability = _handle_old_structure(file)
            else:
                tmp = _handle_old_structure(file)
                availability = pd.concat([availability, tmp])
        except FileNotFoundError:
            logging.info(
                "\tNo .StationAvailability file found for "
                f"{readstart.year} - {readstart.julday:03d}"
            )
        readstart += 86400

    if availability is None:
        raise NoStationAvailabilityData(fpath, starttime, endtime)

    starttime, endtime = availability.index[0], availability.index[-1]
    logging.debug(f"\t\t...from {starttime} - {endtime}")

    return availability


def _handle_old_structure(
    availability_in_file: pathlib.Path, permanent_conversion: bool = False
) -> pd.DataFrame:
    """
    A short utility function that dynamically converts the old style availability files
    (with column names simply the station) to the new style (with separate columns for
    each station/phase combination). This uses the knowledge that an availability of '1'
    in the old style meant that all data was available (e.g., <station>_P and
    <station>_S available).

    This is only done if the file was in the old format - otherwise it just returns the
    original, unaltered dataframe.

    Parameters
    ----------
    availability_in_file:
        An availability file to be read in, tested and potentially converted.
    permanent_conversion:
        If toggled, the availability file will be permanently converted to the new file
        structure.

    Returns
    -------
    availability_out:
        The corrected (if necessary) availability dataframe.

    """

    availability_in = pd.read_csv(availability_in_file, index_col=0)

    cols = [col_names.split("_") for col_names in availability_in.columns]

    # Check if station + phase are already in the column names
    if len(cols[0]) == 2:
        return availability_in

    availability_out = pd.DataFrame()
    logging.info(
        "\t\tWarning: an availability file is in the old format - converting..."
    )
    for phase in "PS":
        for stat in cols:
            new_key = f"{stat[0]}_{phase}"
            availability_out[new_key] = availability_in[stat[0]].values
    availability_out.index = availability_in.index

    if permanent_conversion:
        availability_out.to_csv(availability_in_file)

    return availability_out


def write_availability(run: Run, availability: pd.DataFrame) -> None:
    """
    Write out csv files (split by Julian day) containing station availability data.

    If a file for the target year/Julian day already exists, append to it rather than
    overwriting it. This supports runs split into multiple sub-day chunks.

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    availability:
        Details the availability of each station for each timestep of detect.

    """

    fpath = run.path / "detect" / "availability"
    fpath.mkdir(exist_ok=True, parents=True)

    availability.index = times = pd.to_datetime(availability.index)
    datelist = set([time.date() for time in times])

    for date in datelist:
        to_write = availability[availability.index.date == date]
        date = UTCDateTime(date)

        fstem = f"{date.year}_{date.julday:03d}_StationAvailability"
        file = (fpath / fstem).with_suffix(".csv")

        if file.exists():
            existing = _handle_old_structure(file)
            existing.index = pd.to_datetime(existing.index)
            to_write = pd.concat([existing, to_write])
            to_write = to_write[~to_write.index.duplicated(keep="last")]
            to_write = to_write.sort_index()

        to_write.index = [UTCDateTime(idx) for idx in to_write.index]

        to_write.to_csv(file)
