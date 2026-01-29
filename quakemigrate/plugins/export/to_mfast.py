"""
This module provides parsers to generate SAC waveform files from an ObsPy Catalog, with
headers correctly populated for MFAST.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from obspy import read
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth


if TYPE_CHECKING:
    from obspy.core.event import Event

    from quakemigrate.io.station import Station


cmpaz = {"N": 0, "Z": 0, "E": 90}
cmpinc = {"N": 90, "Z": 0, "E": 90}


def sac_mfast(
    event: Event,
    stations: list[Station],
    output_path: str,
    units: str,
    filename: str | None = None,
) -> None:
    """
    Function to create the SAC file.

    Parameters
    ----------
    event:
        Contains information about the origin time and a list of associated picks.
    stations:
        List of Station objects containing station information.
    output_path:
        Location to save the SAC file.
    units:
        Grid projection coordinates for QM LUT (determines units of depths and
        uncertainties in the .event files).
    filename:
        Name of SAC file - defaults to "eventid/eventid.stationid.{comp}".

    Raises
    ------
    ValueError
        If an invalid value of `unit` has been supplied.

    """

    # Read in the mSEED file containing
    stream = read(event.extra.cut_waveforms_file.value)

    # Set distance conversion factor (from units of QM LUT projection units).
    if units == "km":
        factor = 1
    elif units == "m":
        factor = 1e3
    else:
        raise ValueError(f"units must be 'km' or 'm'; not {units}")

    # Create general SAC header AttribDict
    event_header = AttribDict()
    origin = event.preferred_origin()
    event_header.evla = origin.latitude
    event_header.evlo = origin.longitude
    # Obspy Event object already has all units converted to metres
    event_header.evdp = origin.depth / factor  # converted to km
    eventid = str(event.resource_id)
    if filename is None:
        filename = eventid + ".{}.{}"
    else:
        filename = filename + ".{}.{}"
    output_path = pathlib.Path(output_path) / eventid
    output_path.mkdir(parents=True, exist_ok=True)

    # Loop over the available stations and get the pick information
    for station in stations:
        st = stream.select(station=station.station)

        station_header = AttribDict()
        station_header.stla = station.latitude
        station_header.stlo = station.longitude
        station_header.stel = station.elevation / factor  # convert to m

        # Calculate the distance and azimuth between event and station
        dist, az, _ = gps2dist_azimuth(
            origin.latitude, origin.longitude, station.latitude, station.longitude
        )

        station_header.dist = dist / 1000.0  # convert m to km
        station_header.az = az

        # Get relevant picks here
        picks = []
        for pick in event.picks:
            if pick.waveform_id.station_code == station.id:
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
        pick_header.kt5 = str(kt5)
        pick_header.kt0 = str(kt5)
        pick_header.o = origin_time
        if p_pick != 0:
            pick_header.a = p_pick

        for comp in ["Z", "N", "E"]:
            tr = st.select(channel=f"*{comp}")[0]

            # Write out to SAC file, then read in again to fill header
            fname = filename.format(station.id, comp.lower())
            tr.write(output_path / fname, format="SAC")
            tr = read(output_path / fname)[0]

            sac_header = AttribDict()
            sac_header.cmpaz = str(cmpaz[comp])
            sac_header.cmpinc = str(cmpinc[comp])
            sac_header.kcmpnm = f"HH{comp}"
            sac_header.update(event_header)
            sac_header.update(station_header)
            sac_header.update(pick_header)
            tr.stats.sac.update(sac_header)
            tr.write(output_path / fname, format="SAC")
