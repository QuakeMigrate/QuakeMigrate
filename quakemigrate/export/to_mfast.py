# -*- coding: utf-8 -*-
"""
This module provides parsers to generate SAC waveform files from an ObsPy
Catalog, with headers correctly populated for MFAST.

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib

from obspy import read
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth


cmpaz = {"N": 0, "Z": 0, "E": 90}
cmpinc = {"N": 90, "Z": 0, "E": 90}


def sac_mfast(event, stations, output_path, filename=None):
    """
    Function to create the SAC file.

    Parameters
    ----------
    event : `ObsPy.Event` object
        Contains information about the origin time and a list of associated
        picks.
    stations : `pandas.DataFrame` object
        DataFrame containing station information.
    output_path : str
        Location to save the SAC file.
    filename : str, optional
        Name of SAC file - defaults to "eventid/eventid.station.{comp}".

    """

    # Read in the mSEED file containing
    stream = read(event.event_descriptions.text)

    # Create general SAC header AttribDict
    event_header = AttribDict()
    origin = event.preferred_origin()
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
