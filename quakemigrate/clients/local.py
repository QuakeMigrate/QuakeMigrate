"""
Local archive waveform client.

Provides a waveform client for miniSEED or other ObsPy-readable waveform data stored
in a local filesystem archive with a regular directory and/or filename pattern.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import pathlib
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING

from obspy import read, Stream, UTCDateTime

from quakemigrate.exceptions import ArchiveEmpty, InvalidArchivePathStructure
from .base import BaseWaveformClient


if TYPE_CHECKING:
    from collections.abc import Iterator

    from quakemigrate.io.station import Station


_ALLOWED_FIELDS = {
    "year",
    "month",
    "day",
    "jday",
    "network",
    "station",
    "location",
    "channels",
    "dtime",
}


def _root_field(field_name: str) -> str:
    """
    Return the root field for a format field name.

    Parameters
    ----------
    field_name:
        Format field name extracted from a Python format string.

    Returns
    -------
    root_field:
        The root field, e.g., `dtime` from `dtime.year`.

    """

    i = field_name.find(".")
    if i != -1:
        return field_name[:i]

    return field_name


def _validate_path_format(archive_format: str) -> str:
    """
    Validate an archive path format string.

    Ensures that every placeholder used in the format string is supported by
    :meth:`Archive._load_from_path`.

    Supported placeholders are:

    - year
    - month
    - day
    - jday
    - network
    - station
    - location
    - channels
    - dtime

    Parameters
    ----------
    archive_format:
        The template archive path string to be formatted during data queries.

    Returns
    -------
    archive_format:
        The template archive path string to be formatted during data queries.

    Raises
    ------
    InvalidArchivePathStructure
        Raised if there is a placeholder that is not supported.

    """

    allowed = set(_ALLOWED_FIELDS)

    for _, field_name, _, _ in string.Formatter().parse(archive_format):
        if field_name is None:
            continue

        if field_name == "":
            raise InvalidArchivePathStructure(archive_format)

        root = _root_field(field_name)
        if root not in allowed:
            raise InvalidArchivePathStructure(root)

    return archive_format


@dataclass(kw_only=True)
class LocalWaveformClient(BaseWaveformClient):
    """
    Waveform client for a local filesystem archive.

    This client reads waveform files directly from a local archive using a user-defined
    path template. The template is evaluated once per station per day over the requested
    time range, and all matching files are read with ObsPy.

    Archive inherits from :class:`~quakemigrate.clients.base.BaseWaveformClient` and
    implements the abstract `_fetch_stream` method.

    Parameters
    ----------
    path:
        Root directory of the waveform archive.
    format:
        Archive path template, relative to path. Supported placeholders are specified in
        :func:`_validate_path_format`.

    """

    path: str | pathlib.Path
    format: str

    def __post_init__(self) -> None:
        """Cast and validate input arguments."""

        super().__post_init__()
        self.path = pathlib.Path(self.path)
        self.format = _validate_path_format(self.format)

    def _fetch_stream(
        self, stations: list[Station], starttime: UTCDateTime, endtime: UTCDateTime
    ) -> Stream:
        """
        Fetch waveform data from the local waveform archive.

        Matching files are identified from the configured archive path template and read
        with ObsPy. All readable traces found within the requested time range are
        returned in a single stream.

        Parameters
        ----------
        stations:
            List of stations for which waveform data should be loaded.
        starttime:
            First timestamp of data to be loaded from the local archive.
        endtime:
            Final timestamp of data to be loaded from the local archive.

        Returns
        -------
        st:
            Stream containing all readable waveform data found in the local archive.

        Raises
        ------
        ArchiveEmpty
            Raised if no matching files are found in the archive for the request.

        """

        files = self._load_from_path(stations, starttime, endtime)

        st = Stream()
        try:
            first = next(files)
        except StopIteration:
            raise ArchiveEmpty()

        st += self._read_file(first, starttime=starttime, endtime=endtime)
        for file in files:
            st += self._read_file(file, starttime=starttime, endtime=endtime)

        return st

    def _read_file(
        self, file: pathlib.Path, starttime: UTCDateTime, endtime: UTCDateTime
    ) -> Stream:
        """
        Read a waveform file for a requested time window.

        Parameters
        ----------
        file:
            Path to the waveform file.
        starttime:
            Start of the requested time window.
        endtime:
            End of the requested time window.

        Returns
        -------
        st:
            Stream read from the file. If the file is not compatible with ObsPy, an
            empty stream is returned and the file is skipped.

        """

        try:
            return read(file, starttime=starttime, endtime=endtime, nearest_sample=True)
        except TypeError:
            logging.info(f"File not compatible with ObsPy - {file}")
            return Stream()

    def _load_from_path(
        self, stations: list[Station], starttime: UTCDateTime, endtime: UTCDateTime
    ) -> Iterator[pathlib.Path]:
        """
        Yield archive files matching a station/time request.

        The archive template is expanded once per station per day between starttime and
        endtime inclusive. Any matching filesystem paths are yielded in the order they
        are discovered.

        Parameters
        ----------
        stations:
            List of stations to search for in the archive.
        starttime:
            Start of the requested time window.
        endtime:
            End of the requested time window.

        Yields
        ------
        pathlib.Path
            Paths to waveform files matching the archive template.

        """

        loadstart = UTCDateTime(starttime.date)
        while loadstart <= endtime:
            for station in stations:
                file_format = self.format.format(
                    year=loadstart.year,
                    month=loadstart.month,
                    day=loadstart.day,
                    jday=loadstart.julday,
                    network=station.network if station.network is not None else "*",
                    station=station.station,
                    location=station.location if station.location is not None else "*",
                    channels=station.channels if station.channels is not None else "*",
                    dtime=loadstart,
                )
                for file in self.path.glob(file_format):
                    yield file
            loadstart += 86400

    def _client_description(self) -> list[str]:
        return [
            f"\tArchive root:\t{self.path}",
            f"\tPath format:\t{self.format}",
        ]
