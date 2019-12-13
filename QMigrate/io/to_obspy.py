# -*- coding: utf-8 -*-
"""
This module provides parsers to export the output of a QuakeMigrate run to an
ObsPy Catalog.

"""

from itertools import chain
import pathlib

import pandas as pd

from obspy import Catalog, UTCDateTime, __version__
from obspy.core import AttribDict
from obspy.core.event import (Arrival, Event, Origin, Pick, EventDescription,
                              WaveformStreamID)
from obspy.geodetics import kilometer2degrees


def read_quakemigrate(run_dir):
    """
    Reads the .event and .picks outputs from a QuakeMigrate run into an obspy
    Catalog object

    Parameters
    ----------
    run_dir: str
        Path to QuakeMigrate run directory

    Returns
    -------
    catalog : obspy Catalog object

    """

    run_dir = pathlib.Path(run_dir)

    if run_dir.is_dir():
        try:
            events = run_dir.glob("locate/events/*.event")
            first = next(events)
            events = chain([first], events)
        except StopIteration:
            pass

    cat = Catalog()

    for event in events:
        event = _read_single_event(event)
        if event is None:
            continue
        else:
            cat.append(event)

    cat.creation_info.creation_time = UTCDateTime()
    cat.creation_info.version = "ObsPy %s" % __version__
    return cat


def _read_single_event(event_file):
    """
    Parse an event file from QuakeMigrate into an obspy Event object

    """

    # Parse information from event file
    event_info = pd.read_csv(event_file).iloc[0]

    # Create event object to store origin and pick information
    event = Event()
    event.resource_id = event_file.stem

    # Determine location of cut stream data
    mseed = event_file.parents[1] / "cut_waveforms" / "{}".format(event_file.stem)
    event_description = EventDescription(str(mseed.with_suffix(".m")))
    event.event_descriptions = event_description

    # Create origin with location + uncertainty and associate with event
    o = Origin()
    o.longitude = event_info["X"]
    o.latitude = event_info["Y"]
    o.depth = event_info["Z"]
    o.time = UTCDateTime(event_info["DT"])
    event.origins = [o]
    event.preferred_origin_id = o.resource_id

    o.longitude_errors.uncertainty = kilometer2degrees(
        event_info["LocalGaussian_ErrX"] / 1e3)
    o.latitude_errors.uncertainty = kilometer2degrees(
        event_info["LocalGaussian_ErrY"] / 1e3)
    o.depth_errors.uncertainty = event_info["LocalGaussian_ErrZ"]

    # --- Handle pick files ---
    pick_file = event_file.parents[1] / "picks" / "{}".format(event_file.stem)
    if pick_file.with_suffix(".picks").is_file():
        picks = pd.read_csv(pick_file.with_suffix(".picks"))
    else:
        return None

    for i, p in picks.iterrows():
        arrival = Arrival()
        o.arrivals.append(arrival)
        station = str(p["Name"])
        phase = str(p["Phase"])
        arrival.phase = phase
        # arrival.distance = kilometer2degrees(float(line[21]))
        # arrival.azimuth = float(line[23])
        pick = Pick()

        wid = WaveformStreamID(network_code="", station_code=station)

        if str(p["PickTime"]) != "-1":
            t = UTCDateTime(p["PickTime"])
            pick.waveform_id = wid
            pick.method_id = "QuakeMigrate"
            pick.time = t
            pick.time_errors.uncertainty = float(p["PickError"])
            pick.phase_hint = phase

            event.picks.append(pick)
            arrival.pick_id = pick.resource_id

    return event
