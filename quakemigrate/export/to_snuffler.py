# -*- coding:utf-8 -*-
"""
This module provides parsers to generate input files for Snuffler, a manual phase
picking interface from the Pyrocko package.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib


def snuffler_stations(stations, output_path, filename, network_code=None):
    """
    Function to create station files compatible with snuffler.

    Parameters
    ----------
    stations : `pandas.DataFrame` object
        DataFrame containing station information.
    output_path : str
        Location to save snuffler station file.
    filename : str
        Name of output station file.
    network_code : str
        Unique identifier for the seismic network.

    """

    output = pathlib.Path(output_path) / filename

    line_template = "{nw}.{stat}. {lat} {lon} {elev} {dep}\n"

    with output.open(mode="w") as f:
        for i, station in stations.iterrows():
            if network_code is None:
                try:
                    network_code = station["Network"]
                except KeyError:
                    network_code = ""

            line = line_template.format(
                nw=network_code,
                stat=station["Name"],
                lat=station["Latitude"],
                lon=station["Longitude"],
                elev=station["Elevation"],
                dep="0",
            )

            f.write(line)


def snuffler_markers(event, output_path, filename=None):
    """
    Function to create marker files compatible with snuffler

    Parameters
    ----------
    event : `ObsPy.Event` object
        Contains information about the origin time and a list of associated picks.
    output_path : str
        Location to save the marker file.
    filename : str, optional
        Name of marker file - defaults to 'eventid/eventid.markers'.

    """

    if filename is None:
        filename = f"{event.resource_id}.markers"

    output_path = pathlib.Path(output_path) / str(event.resource_id)
    output_path.mkdir(parents=True, exist_ok=True)

    output = output_path / filename

    line_template = (
        "phase: {year}-{month}-{day} {hr}:{min}:{sec}.{msec} 5 {nw}.{stat}..{comp} "
        "None None None {phase} None False\n"
    )

    origin = event.origins[0]

    # Write event line to file
    event_line = (
        "event: {year}-{month}-{day} {hr}:{min}:{sec}.{msec} 0 {eventid} 0.0 0.0 None "
        "None None Event None\n"
    )

    event_line = event_line.format(
        year=origin.time.year,
        month=str(origin.time.month).zfill(2),
        day=str(origin.time.day).zfill(2),
        hr=str(origin.time.hour).zfill(2),
        min=str(origin.time.minute).zfill(2),
        sec=str(origin.time.second).zfill(2),
        msec=origin.time.microsecond,
        eventid=str(event.resource_id),
    )

    with output.open("w") as f:
        f.write("# Snuffler Markers File Version 0.2\n")
        f.write(event_line)

        for pick in event.picks:
            if pick.phase_hint == "P":
                comp = "BHZ"
            elif pick.phase_hint == "S":
                comp = "BHN"
            line = line_template.format(
                year=pick.time.year,
                month=str(pick.time.month).zfill(2),
                day=str(pick.time.day).zfill(2),
                hr=str(pick.time.hour).zfill(2),
                min=str(pick.time.minute).zfill(2),
                sec=str(pick.time.second).zfill(2),
                msec=pick.time.microsecond,
                nw=pick.waveform_id.network_code,
                stat=pick.waveform_id.station_code,
                comp=comp,
                phase=pick.phase_hint,
            )

            f.write(line)
