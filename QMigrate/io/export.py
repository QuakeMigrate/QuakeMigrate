# -*- coding:utf-8 -*-
"""
Generate SAC waveform files from an Obspy Catalogue, with headers correctly
populated for MFAST.

Author: Conor Bacon
Date: 10/10/2019
"""

import pathlib
import warnings

from obspy import read
from obspy.core import AttribDict
from obspy.core.event import Event
from obspy.geodetics import gps2dist_azimuth


cmpaz = {"N": 0, "Z": 0, "E": 90}
cmpinc = {"N": 90, "Z": 0, "E": 90}


def sac_mfast(event, stations, output_path, filename=None):
    """
    Function to create the SAC file

    Parameters
    ----------
    event : ObsPy Event object
        Contains information about the origin time and a list of associated
        picks

    stations : pandas DataFrame
        DataFrame containing station information

    output_path : str
        Location to save the SAC file

    filename : str, optional
        Name of SAC file - defaults to eventid/eventid.station.{comp}

    """

    # Read in the mSEED file containing
    stream = read(event.event_descriptions.text)

    # Create general SAC header AttribDict
    event_header = AttribDict()
    origin = event.origins[0]
    event_header.evla = origin.latitude
    event_header.evlo = origin.longitude
    event_header.evdp = origin.depth / 1000.  # converted to km
    eventid = str(event.resource_id)
    if filename is None:
        filename = eventid + ".{}.{}"
    else:
        filename = filename + ".{}.{}"
    output_path = pathlib.Path(output_path) / eventid
    output_path.mkdir(parents=True, exist_ok=True)

    # Loop over the available stations and get the pick information
    for i, station in stations.iterrows():
        st = stream.select(station=station.Name)

        station_header = AttribDict()
        station_header.stla = station.Latitude
        station_header.stlo = station.Longitude
        station_header.stel = station.Elevation

        # Calculate the distance and azimuth between event and station
        dist, az, baz = gps2dist_azimuth(event_header.evla,
                                         event_header.evlo,
                                         station.Latitude,
                                         station.Longitude)

        station_header.dist = dist / 1000.
        station_header.az = az

        # Get relevant picks here
        picks = []
        for pick in event.picks:
            if pick.waveform_id.station_code == station.Name:
                picks.append(pick)

        if not picks:
            # If no phase picks for this station, continue
            continue

        reference = st[0].stats.starttime
        origin_time = origin.time - reference
        p_pick = s_pick = 0
        for pick in picks:
            if pick.phase_hint == "P":
                p_pick = pick.time - reference
            elif pick.phase_hint == "S":
                s_pick = pick.time - reference

        if s_pick == 0:
            continue

        # Set pick error (think about good bounds?)
        kt5 = 1

        pick_header = AttribDict()
        pick_header.t0 = s_pick
        pick_header.kt5 = kt5
        pick_header.kt0 = kt5
        pick_header.o = origin_time
        if p_pick != 0:
            pick_header.a = p_pick

        for comp in ["Z", "N", "E"]:
            tr = st.select(channel="*{}".format(comp))[0]

            # Write out to SAC file, then read in again to fill header
            name = filename.format(station.Name, comp.lower())
            name = str(output_path / name)
            tr.write(name, format="SAC")
            tr = read(name)[0]

            sac_header = AttribDict()
            sac_header.cmpaz = str(cmpaz[comp])
            sac_header.cmpinc = str(cmpinc[comp])
            sac_header.kcmpnm = "HH{}".format(comp)
            sac_header.update(event_header)
            sac_header.update(station_header)
            sac_header.update(pick_header)
            tr.stats.sac.update(sac_header)
            tr.write(name, format="SAC")


def snuffler_stations(stations, output_path, filename, network_code=None):
    """
    Function to create station files compatible with snuffler.

    Parameters
    ----------
    stations : pandas DataFrame
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

            line = line_template.format(nw=network_code,
                                        stat=station["Name"],
                                        lat=station["Latitude"],
                                        lon=station["Longitude"],
                                        elev=station["Elevation"],
                                        dep="0")

            f.write(line)


def snuffler_markers(event, stations, output_path, filename=None):
    """
    Function to create marker files compatible with snuffler

    Parameters
    ----------
    event : ObsPy Event object
        Contains information about the origin time and a list of associated
        picks

    stations : pandas DataFrame
        DataFrame containing station information

    output_path : str
        Location to save the marker file

    filename : str, optional
        Name of marker file - defaults to eventid/eventid.markers

    """

    if filename is None:
        filename = "{0}.markers".format(str(event.resource_id))

    output_path = pathlib.Path(output_path) / str(event.resource_id)
    output_path.mkdir(parents=True, exist_ok=True)

    output = output_path / filename

    line_template = "phase: {year}-{month}-{day} {hr}:{min}:{sec}.{msec} 5 "
    line_template += "{nw}.{stat}..{comp} None None None {phase} None False\n"

    origin = event.origins[0]

    # Write event line to file
    event_line = "event: {year}-{month}-{day} {hr}:{min}:{sec}.{msec} 0 "
    event_line += "{eventid} 0.0 0.0 None None None Event None\n"

    event_line = event_line.format(year=origin.time.year,
                                   month=str(origin.time.month).zfill(2),
                                   day=str(origin.time.day).zfill(2),
                                   hr=str(origin.time.hour).zfill(2),
                                   min=str(origin.time.minute).zfill(2),
                                   sec=str(origin.time.second).zfill(2),
                                   msec=origin.time.microsecond,
                                   eventid=str(event.resource_id))

    with output.open("w") as f:
        f.write("# Snuffler Markers File Version 0.2\n")
        f.write(event_line)

        for pick in event.picks:
            if pick.phase_hint == "P":
                comp = "BHZ"
            elif pick.phase_hint == "S":
                comp = "BHN"
            line = line_template.format(year=pick.time.year,
                                        month=str(pick.time.month).zfill(2),
                                        day=str(pick.time.day).zfill(2),
                                        hr=str(pick.time.hour).zfill(2),
                                        min=str(pick.time.minute).zfill(2),
                                        sec=str(pick.time.second).zfill(2),
                                        msec=pick.time.microsecond,
                                        nw=pick.waveform_id.network_code,
                                        stat=pick.waveform_id.station_code,
                                        comp=comp,
                                        phase=pick.phase_hint)

            f.write(line)


ONSETS = {"i": "impulsive", "e": "emergent"}
ONSETS_REVERSE = {"impulsive": "i", "emergent": "e"}
POLARITIES = {"c": "positive", "u": "positive", "d": "negative"}
POLARITIES_REVERSE = {"positive": "u", "negative": "d"}


def nlloc_obs(event, filename):
    """
    Write a NonLinLoc Phase file from an obspy Catalog object.

    Parameters
    ----------
    event : obspy Catalog object
        Contains information on a single event.

    filename : str
        Name of NonLinLoc phase file.

    """
    info = []

    if not isinstance(event, Event):
        msg = ("Writing NonLinLoc Phase file is only supported for Catalogs "
               "with a single Event in it (use a for loop over the catalog "
               "and provide an output file name for each event).")
        raise ValueError(msg)

    fmt = ("{:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} " +
           "{:7.4f} GAU {:9.2e} {:9.2e} {:9.2e} {:9.2e} {:9.2e}")

    for pick in event.picks:
        wid = pick.waveform_id
        station = wid.station_code or "?"
        component = wid.channel_code and wid.channel_code[-1].upper() or "?"
        if component not in "ZNEH":
            component = "?"
        onset = ONSETS_REVERSE.get(pick.onset, "?")
        phase_type = pick.phase_hint or "?"
        polarity = POLARITIES_REVERSE.get(pick.polarity, "?")
        date = pick.time.strftime("%Y%m%d")
        hourminute = pick.time.strftime("%H%M")
        seconds = pick.time.second + pick.time.microsecond * 1e-6
        time_error = pick.time_errors.uncertainty or -1
        if time_error == -1:
            try:
                time_error = (pick.time_errors.upper_uncertainty +
                              pick.time_errors.lower_uncertainty) / 2.0
            except Exception:
                pass
        info_ = fmt.format(station.ljust(6), "?".ljust(4), component.ljust(4),
                           onset.ljust(1), phase_type.ljust(6), polarity.ljust(1),
                           date, hourminute, seconds, time_error, -1, -1, -1, 1)
        info.append(info_)

    if info:
        info = "\n".join(sorted(info) + [""])
    else:
        msg = "No pick information, writing empty NLLOC OBS file."
        warnings.warn(msg)
    with open(filename, "w") as fh:
        for line in info:
            fh.write(line)
