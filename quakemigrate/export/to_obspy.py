# -*- coding: utf-8 -*-
"""
This module provides parsers to export the output of a QuakeMigrate run to an ObsPy
Catalog.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from itertools import chain
import pathlib

import pandas as pd

from obspy import Catalog, UTCDateTime, __version__
from obspy.core import AttribDict
from obspy.core.event import (
    Event,
    Origin,
    OriginUncertainty,
    ConfidenceEllipsoid,
    Pick,
    TimeWindow,
    WaveformStreamID,
    CreationInfo,
    Amplitude,
    StationMagnitude,
    Magnitude,
)
from obspy.geodetics import kilometer2degrees

import quakemigrate


ns = "http://quakemigrate.github.io/xmlns/event"


def read_quakemigrate(run_dir, units, run_subname="", local_mag_ph="S"):
    """
    Reads the .event and .picks outputs, and .amps outputs if available, from a
    QuakeMigrate run into an obspy Catalog object.

    NOTE: if a station_corrections dict was used to calculate the network-averaged local
    magnitude, this information will not be included in the ObsPy event object. There
    might therefore be a discrepancy between the mean of the StationMagnitudes and the
    event magnitude.

    Parameters
    ----------
    run_dir : str
        Path to QuakeMigrate run directory.
    units : {"km", "m"}
        Grid projection coordinates for QM LUT (determines units of depths and
        uncertainties in the .event files).
    run_subname : str, optional
        Run_subname string (if applicable).
    local_mag_ph : {"S", "P"}, optional
        Amplitude measurement used to calculate local magnitudes. (Default "S").

    Returns
    -------
    cat : `obspy.Catalog` object
        Catalog containing events in the specified QuakeMigrate run directory.

    """

    locate_dir = pathlib.Path(run_dir) / "locate" / run_subname
    events_dir = locate_dir / "events"

    if events_dir.is_dir():
        try:
            event_files = events_dir.glob("*.event")
            first = next(event_files)
            event_files = chain([first], event_files)
        except StopIteration:
            pass

    cat = Catalog()

    for eventf in event_files:
        event = _read_single_event(eventf, locate_dir, units, local_mag_ph)
        if event is None:
            continue
        else:
            cat.append(event)

    cat.creation_info.creation_time = UTCDateTime()
    cat.creation_info.version = f"ObsPy {__version__}"

    return cat


def _read_single_event(event_file, locate_dir, units, local_mag_ph):
    """
    Parse an event file from QuakeMigrate into an obspy Event object.

    Parameters
    ----------
    event_file : `pathlib.Path` object
        Path to .event file to read.
    locate_dir : `pathlib.Path` object
        Path to locate directory (contains "events", "picks" etc. directories).
    units : {"km", "m"}
        Grid projection coordinates for QM LUT (determines units of depths and
        uncertainties in the .event files).
    local_mag_ph : {"S", "P"}
        Amplitude measurement used to calculate local magnitudes.

    Returns
    -------
    event : `obspy.Event` object
        Event object populated with all available information output by
        :class:`~quakemigrate.signal.scan.locate()`, including event locations and
        uncertainties, picks, and amplitudes and magnitudes if available.

    """

    # Parse information from event file
    event_info = pd.read_csv(event_file).iloc[0]
    event_uid = str(event_info["EventID"])

    # Set distance conversion factor (from units of QM LUT projection units).
    if units == "km":
        factor = 1e3
    elif units == "m":
        factor = 1
    else:
        raise AttributeError(f"units must be 'km' or 'm'; not {units}")

    # Create event object to store origin and pick information
    event = Event()
    event.extra = AttribDict()
    event.resource_id = str(event_info["EventID"])
    event.creation_info = CreationInfo(
        author="QuakeMigrate", version=quakemigrate.__version__
    )

    # Add COA info to extra
    event.extra.coa = {"value": event_info["COA"], "namespace": ns}
    event.extra.coa_norm = {"value": event_info["COA_NORM"], "namespace": ns}
    event.extra.trig_coa = {"value": event_info["TRIG_COA"], "namespace": ns}
    event.extra.dec_coa = {"value": event_info["DEC_COA"], "namespace": ns}
    event.extra.dec_coa_norm = {"value": event_info["DEC_COA_NORM"], "namespace": ns}

    # Determine location of cut waveform data - add to event object as a
    # custom extra attribute.
    mseed = locate_dir / "raw_cut_waveforms" / event_uid
    event.extra.cut_waveforms_file = {
        "value": str(mseed.with_suffix(".m").resolve()),
        "namespace": ns,
    }
    if (locate_dir / "real_cut_waveforms").exists():
        mseed = locate_dir / "real_cut_waveforms" / event_uid
        event.extra.real_cut_waveforms_file = {
            "value": str(mseed.with_suffix(".m").resolve()),
            "namespace": ns,
        }
    if (locate_dir / "wa_cut_waveforms").exists():
        mseed = locate_dir / "wa_cut_waveforms" / event_uid
        event.extra.wa_cut_waveforms_file = {
            "value": str(mseed.with_suffix(".m").resolve()),
            "namespace": ns,
        }

    # Create origin with spline location and set to preferred event origin.
    origin = Origin()
    origin.method_id = "spline"
    origin.longitude = event_info["X"]
    origin.latitude = event_info["Y"]
    origin.depth = event_info["Z"] * factor
    origin.time = UTCDateTime(event_info["DT"])
    event.origins = [origin]
    event.preferred_origin_id = origin.resource_id

    # Create origin with gaussian location and associate with event
    origin = Origin()
    origin.method_id = "gaussian"
    origin.longitude = event_info["GAU_X"]
    origin.latitude = event_info["GAU_Y"]
    origin.depth = event_info["GAU_Z"] * factor
    origin.time = UTCDateTime(event_info["DT"])
    event.origins.append(origin)

    ouc = OriginUncertainty()
    ce = ConfidenceEllipsoid()
    ce.semi_major_axis_length = event_info["COV_ErrY"] * factor
    ce.semi_intermediate_axis_length = event_info["COV_ErrX"] * factor
    ce.semi_minor_axis_length = event_info["COV_ErrZ"] * factor
    ce.major_axis_plunge = 0
    ce.major_axis_azimuth = 0
    ce.major_axis_rotation = 0
    ouc.confidence_ellipsoid = ce
    ouc.preferred_description = "confidence ellipsoid"

    # Set uncertainties for both as the gaussian uncertainties
    for origin in event.origins:
        origin.longitude_errors.uncertainty = kilometer2degrees(
            event_info["GAU_ErrX"] * factor / 1e3
        )
        origin.latitude_errors.uncertainty = kilometer2degrees(
            event_info["GAU_ErrY"] * factor / 1e3
        )
        origin.depth_errors.uncertainty = event_info["GAU_ErrZ"] * factor
        origin.origin_uncertainty = ouc

    # Add OriginQuality info to each origin?
    for origin in event.origins:
        origin.origin_type = "hypocenter"
        origin.evaluation_mode = "automatic"

    # --- Handle picks file ---
    pick_file = locate_dir / "picks" / event_uid
    if pick_file.with_suffix(".picks").is_file():
        picks = pd.read_csv(pick_file.with_suffix(".picks"))
    else:
        return None

    for _, pickline in picks.iterrows():
        station = str(pickline["Station"])
        phase = str(pickline["Phase"])
        wid = WaveformStreamID(network_code="", station_code=station)

        for method in ["modelled", "autopick"]:
            pick = Pick()
            pick.extra = AttribDict()
            pick.waveform_id = wid
            pick.method_id = method
            pick.phase_hint = phase
            if method == "autopick" and str(pickline["PickTime"]) != "-1":
                pick.time = UTCDateTime(pickline["PickTime"])
                pick.time_errors.uncertainty = float(pickline["PickError"])
                pick.extra.snr = {"value": float(pickline["SNR"]), "namespace": ns}
            elif method == "modelled":
                pick.time = UTCDateTime(pickline["ModelledTime"])
            else:
                continue
            event.picks.append(pick)

    # --- Handle amplitudes file ---
    amps_file = locate_dir / "amplitudes" / event_uid
    if amps_file.with_suffix(".amps").is_file():
        amps = pd.read_csv(amps_file.with_suffix(".amps"))

        i = 0
        for _, ampsline in amps.iterrows():
            wid = WaveformStreamID(seed_string=ampsline["id"])
            noise_amp = ampsline["Noise_amp"] / 1000  # mm to m
            for phase in ["P_amp", "S_amp"]:
                amp = Amplitude()
                if pd.isna(ampsline[phase]):
                    continue
                amp.generic_amplitude = ampsline[phase] / 1000  # mm to m
                amp.generic_amplitude_errors.uncertainty = noise_amp
                amp.unit = "m"
                amp.type = "AML"
                amp.method_id = phase
                amp.period = 1 / ampsline[f"{phase[0]}_freq"]
                amp.time_window = TimeWindow(
                    reference=UTCDateTime(ampsline[f"{phase[0]}_time"])
                )
                # amp.pick_id = ?
                amp.waveform_id = wid
                # amp.filter_id = ?
                amp.magnitude_hint = "ML"
                amp.evaluation_mode = "automatic"
                amp.extra = AttribDict()
                try:
                    amp.extra.filter_gain = {
                        "value": ampsline[f"{phase[0]}_filter_gain"],
                        "namespace": ns,
                    }
                    amp.extra.avg_amp = {
                        "value": ampsline[f"{phase[0]}_avg_amp"] / 1000,  # m
                        "namespace": ns,
                    }
                except KeyError:
                    pass

                if phase[0] == local_mag_ph and not pd.isna(ampsline["ML"]):
                    i += 1
                    stat_mag = StationMagnitude()
                    stat_mag.extra = AttribDict()
                    # stat_mag.origin_id = ? local_mag_loc
                    stat_mag.mag = ampsline["ML"]
                    stat_mag.mag_errors.uncertainty = ampsline["ML_Err"]
                    stat_mag.station_magnitude_type = "ML"
                    stat_mag.amplitude_id = amp.resource_id
                    stat_mag.extra.picked = {
                        "value": ampsline["is_picked"],
                        "namespace": ns,
                    }
                    stat_mag.extra.epi_dist = {
                        "value": ampsline["epi_dist"],
                        "namespace": ns,
                    }
                    stat_mag.extra.z_dist = {
                        "value": ampsline["z_dist"],
                        "namespace": ns,
                    }

                    event.station_magnitudes.append(stat_mag)

                event.amplitudes.append(amp)

        mag = Magnitude()
        mag.extra = AttribDict()
        mag.mag = event_info["ML"]
        mag.mag_errors.uncertainty = event_info["ML_Err"]
        mag.magnitude_type = "ML"
        # mag.origin_id = ?
        mag.station_count = i
        mag.evaluation_mode = "automatic"
        mag.extra.r2 = {"value": event_info["ML_r2"], "namespace": ns}

        event.magnitudes = [mag]
        event.preferred_magnitude_id = mag.resource_id

    return event
