"""
Small module that provides basic waveform simulations routines.

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from quakemigrate.lut import LUT


@dataclass
class Wavelet:
    """Light utility class to encapsulate a Wavelet."""

    sps: int
    time: np.ndarray
    data: np.ndarray

    def project(self, source_polarisation: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Projects the wavelet onto a source polarisation.

        Parameters
        ----------
        source_polarisation: Source polarisation in degrees.

        Returns
        -------
        wavelet_x: Projected x-component of wavelet function.
        wavelet_y: Projected y-component of wavelet function.

        """

        wavelet_x = self.wavelet * np.cos(np.deg2rad(source_polarisation))
        wavelet_y = self.wavelet * np.sin(np.deg2rad(source_polarisation))

        return wavelet_x, wavelet_y


class GaussianDerivativeWavelet(Wavelet):
    def __init__(self, frequency: float, sps: int, half_timespan: float) -> None:
        """Instantiate GaussianDerivativeWavelet object."""

        delta_T = 1 / frequency
        sigma = delta_T / 6
        self.frequency = frequency
        self.sps = sps

        self.time = np.arange(-half_timespan, half_timespan + 1 / sps, 1 / sps)
        data = (
            -self.time
            * np.exp(-(self.time**2) / (2 * sigma**2))
            / (sigma**3 * np.sqrt(2 * np.pi))
        )

        # Roll wavelet so first motion is at ~midpoint of array
        self.data = np.roll(data, int(sps * 0.5 / frequency) + 3) / max(data)


def simulate_waveforms(
    wavelet: Wavelet,
    earthquake_coords: tuple[float, float, float],
    lut: LUT,
    magnitude: int = 1,
    noise: dict | None = None,
    angle_of_incidence: int = 0,
) -> Stream:
    """
    Simulates the waveforms expected for an earthquake within a given LUT.

    Performs simulation in LQT-space (Latitudinal, SV direction, SH direction), before
    rotating onto ZNE based on the ray angles (back-azimuth and inclination).

    Parameters
    ----------
    wavelet: The base wavelet used to represent the waveform for each simulated phase.                                                                 
    earthquake_coords: The lon, lat, and depth of the earthquake.
    lut: A QuakeMigrate traveltime lookup table, used to migrate simulated waveforms.
    magnitude: A local magnitude used to simulate the effect of distance attenuation.
    noise: Gaussian noise scaling for simulated waveform traveltimes and amplitudes.
    angle_of_incidence: Used to rotate from LQT onto ZNE axes.

    Returns
    -------
    stream: An ObsPy Stream object containing the simulated waveform traces.

    """

    if noise is None:
        noise = {
            "traveltime": {"P": 0.02, "S": 0.02},
            "amplitude": {"P": 0.1, "S": 0.1},
        }

    inclination = 90 - angle_of_incidence
    earthquake_ijk = lut.index2coord(earthquake_coords, inverse=True)

    stream = Stream()
    # Loop over each station and construct the P and S synthetics
    for i, station_data in lut.station_data.iterrows():
        station = station_data["Name"]
        hypo_dist, az, baz = _gps2hypodist_az_baz(
            station_data, earthquake_coords, lut.unit_conversion_factor
        )
        # amp_factor = 1.
        amp_factor = 10 ** (magnitude - _attenuate(hypo_dist))

        # Build L component, e.g. the P-phase synthetic
        P = Trace()
        P_ttime = lut.traveltime_to("P", earthquake_ijk, station=station)
        P_ttime += np.random.normal(scale=noise["traveltime"]["P"], size=1)
        roll_by = int(wavelet.sps * P_ttime)
        P_amp_noise = np.random.normal(
            scale=noise["amplitude"]["P"], size=len(wavelet.data)
        )
        P.data = np.roll(wavelet.data.copy() * amp_factor * 0.5 + P_amp_noise, roll_by)

        # Build Q/T components, e.g. the S-phase synthetic
        S1, S2 = Trace(), Trace()
        S_ttime = lut.traveltime_to("S", earthquake_ijk, station=station)
        S_ttime += np.random.normal(scale=noise["traveltime"]["S"], size=1)
        roll_by = int(wavelet.sps * S_ttime)
        S_amp_noise = np.random.normal(
            scale=noise["amplitude"]["S"], size=len(wavelet.data)
        )
        S1.data = np.roll(wavelet.data.copy() * amp_factor + S_amp_noise, roll_by)
        S2.data = np.zeros(len(S1.data)) + S_amp_noise

        lqt_stream = Stream()
        for component, trace in zip("LQT", [P, S1, S2]):
            trace.stats.starttime = UTCDateTime("2021-02-18T12:00:00.0")
            trace.stats.sampling_rate = wavelet.sps
            trace.stats.station = station
            trace.stats.network = "SC"
            trace.stats.channel = f"CH{component}"
            lqt_stream += trace

        zne_stream = lqt_stream.rotate(
            "LQT->ZNE", back_azimuth=baz, inclination=inclination
        )

        stream += zne_stream

    return stream


def _gps2hypodist_az_baz(
    station_data: pd.DataFrame,
    earthquake_coords: tuple[float, float, float],
    unit_conversion_factor: float,
) -> tuple[float, float, float]:
    """
    Compute the distance between the earthquake hypocentre and the receiver, as well as
    the azimuth/back-azimuth between them.

    Parameters
    ----------
    station_data: DataFrame containing the receiver latitude, longitude, and elevation.
    earthquake_coords: Longitude, latitude, and depth of the earthquake.
    unit_conversion_factor: Factor to convert distances to km.

    Returns
    -------
    hypo_dist: Distance from the hypocentre to the source.
    az: Azimuth from the source to the receiver.
    baz: Back-azimuth from the receiver to the source.

    """

    stla, stlo, stel = station_data[["Latitude", "Longitude", "Elevation"]].values
    evlo, evla, evdp = earthquake_coords

    # Evaluate epicentral distance between station and event.
    # gps2dist_azimuth returns distances in metres -- magnitudes
    # calculation requires distances in kilometres.
    dist, az, baz = gps2dist_azimuth(evla, evlo, stla, stlo)
    epi_dist = dist / 1000

    # Evaluate vertical distance between station and event. Convert to
    # kilometres.
    km_cf = 1000 / unit_conversion_factor
    z_dist = (evdp - stel) / km_cf  # NOTE: stel is actually depth.

    hypo_dist = np.sqrt(z_dist**2 + epi_dist**2)

    return hypo_dist, az, baz


def _attenuate(distance: float) -> float:
    """
    Simulate amplitude attenuation as a function of distance, based on the local
    magnitude scaling equation.

    Parameters
    ----------
    distance: Distance between source and receiver.

    Returns
    -------
    attenuation_factor: Scaling factor as a function of distance.

    """

    return 1.11 * np.log10(distance / 100.0) + 0.00189 * (distance - 100.0) + 3.0
