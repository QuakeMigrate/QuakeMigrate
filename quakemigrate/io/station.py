"""
Station model and helper functions for constructing Station objects from both .csv files
and ObsPy Inventory objects.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import pandas as pd

from quakemigrate.exceptions import InvalidStationFileHeader


if TYPE_CHECKING:
    from obspy.core.inventory import Inventory


@dataclass(slots=True)
class Station:
    station: str

    longitude: float
    latitude: float
    elevation: float
    depth: float

    network: str | None = None
    location: str | None = None
    channels: str | None = None

    read_only: bool = False

    def __str__(self) -> str:
        """Partial SEED ID representation of the station."""
        return ".".join(
            [
                self.network if self.network is not None else "",
                self.station,
                self.location if self.location is not None else "",
            ]
        )

    @property
    def id(self) -> str:
        return str(self)


def stations_from_inventory(
    inventory: Inventory, units: Literal["m", "km"] = "m"
) -> list[Station]:
    """
    Convert an ObsPy Inventory into a list of Station objects.

    Iterates over all networks and stations in the provided Inventory and constructs a
    corresponding list of :class:`Station` objects. Channel and location codes are
    aggregated per station and stored as comma-separated strings.

    Parameters
    ----------
    inventory:
        ObsPy Inventory object containing network and station metadata.
    units:
        Specify the output units. StationXML elevations are defined in metres.

    Returns
    -------
    stations:
        List of Station objects populated with metadata from the Inventory.

    """

    if units not in {"m", "km"}:
        raise ValueError("units must be either 'm' or 'km'")
    unit_conversion_factor = 1.0 if units == "m" else 1_000.0

    stations = []
    for network in inventory:
        for station in network:
            locations, channels = set(), set()
            for channel in station.channels:
                if channel.location_code:
                    locations.add(channel.location_code)
                channels.add(channel.code)

            stations.append(
                Station(
                    network=network.code,
                    station=station.code,
                    location=",".join(sorted(locations)) if locations else None,
                    channels=",".join(sorted(channels)) if channels else None,
                    longitude=station.longitude,
                    latitude=station.latitude,
                    elevation=station.elevation / unit_conversion_factor,
                    depth=-1.0 * station.elevation / unit_conversion_factor,
                )
            )

    return sorted(stations, key=lambda s: s.id)


def read_stations(station_file: str, **kwargs) -> list[Station]:
    """
    Reads station information from file.

    File format (header line is REQUIRED, case sensitive, any order):
        Latitude, Longitude, Elevation, Name
    Optional additional columns:
        Network, Location, Channels

    Note: The units of the station Elevations must match the LUT grid projection.

    Parameters
    ----------
    station_file:
        Path to pandas-readable station file.
    kwargs:
        Passthrough for `pandas.read_csv` kwargs.

    Returns
    -------
    stations:
        List of Station objects.

    Raises
    ------
    InvalidStationFileHeader
        Raised if the input file is missing required entries in the header.

    """

    stations_df = pd.read_csv(station_file, **kwargs)

    try:
        stations = sorted(
            [
                Station(
                    network=station.get("Network"),
                    station=station.Name,
                    location=station.get("Location"),
                    channels=station.get("Channels"),
                    longitude=station.Longitude,
                    latitude=station.Latitude,
                    elevation=station.Elevation,
                    depth=-1.0 * station.Elevation,
                )
                for _, station in stations_df.iterrows()
            ],
            key=lambda s: s.id,
        )
    except AttributeError as e:
        raise InvalidStationFileHeader() from e

    return stations
