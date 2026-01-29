"""
FDSN waveform client.

Provides a QuakeMigrate waveform client backed by the ObsPy
:class:`obspy.clients.fdsn.Client` for retrieving waveform data from a remote FDSN
web service.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from obspy.clients.fdsn import Client

from .base import BaseWaveformClient


if TYPE_CHECKING:
    from obspy import Stream, UTCDateTime

    from quakemigrate.io.station import Station


@dataclass(kw_only=True)
class FDSNWaveformClient(BaseWaveformClient):
    """
    Waveform client backed by a remote FDSN web service.

    This client implements the QuakeMigrate waveform client interface using ObsPy's
    :class:`~obspy.clients.fdsn.Client`. It is responsible only for fetching waveform
    data from the remote service; common waveform post-processing is handled by
    :class:`~quakemigrate.io.clients.base.BaseWaveformClient`.

    Parameters
    ----------
    base_url:
        Base URL or known service name for the FDSN web service.
    timeout:
        Request timeout, in seconds, for FDSN service calls.

    """

    base_url: str
    timeout: int = 60
    _client: Client = field(init=False)

    def __post_init__(self) -> None:
        """Initialise the underlying ObsPy FDSN client."""

        super().__post_init__()
        self._client = Client(self.base_url, timeout=self.timeout)

    def _fetch_stream(
        self, stations: list[Station], starttime: UTCDateTime, endtime: UTCDateTime
    ) -> Stream:
        """
        Fetch waveform data from a remote FDSN station service.

        Parameters
        ----------
        stations:
            List of Station objects for which to request waveform data.
        starttime:
            First timestamp of data to be loaded from the remote FDSN server.
        endtime:
            Final timestamp of data to be loaded from the remote FDSN server.

        Returns
        -------
        st:
            Stream containing the data that has been loaded from the remote FDSN server.

        """

        networks = (
            ",".join(sorted({s.network for s in stations if s.network}))
            if any(s.network for s in stations)
            else "*"
        )
        stations_ = ",".join(sorted({s.station for s in stations}))
        locations = (
            ",".join(sorted({s.location for s in stations if s.location}))
            if any(s.location for s in stations)
            else "*"
        )
        channels = "*"

        return self._client.get_waveforms(
            network=networks,
            station=stations_,
            location=locations,
            channel=channels,
            starttime=starttime,
            endtime=endtime,
        )

    def _client_description(self) -> list[str]:
        return [
            f"\tFDSN service:\t{self.base_url}",
            f"\tTimeout:\t{self.timeout}s",
        ]
