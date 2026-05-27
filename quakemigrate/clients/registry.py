"""
Registry and entry-point discovery for waveform clients.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

from quakemigrate.exceptions import ConfigError
from quakemigrate.clients.base import BaseWaveformClient
from quakemigrate.clients.fdsn import FDSNWaveformClient
from quakemigrate.clients.local import LocalWaveformClient


ENTRY_POINT_GROUP = "quakemigrate.waveform_clients"


_BUILTIN_WAVEFORM_CLIENTS: dict[str, type[BaseWaveformClient]] = {
    "fdsn": FDSNWaveformClient,
    "local": LocalWaveformClient,
}


def _entry_points() -> dict[str, EntryPoint]:
    """Return installed waveform clients plugin entry points."""

    return {
        entry_point.name: entry_point
        for entry_point in entry_points(group=ENTRY_POINT_GROUP)
    }


def list_waveform_clients() -> list[str]:
    """Return the names of built-in and installed waveform clients."""

    return sorted([*_BUILTIN_WAVEFORM_CLIENTS, *_entry_points()])


def get_waveform_client_class(name: str) -> type[BaseWaveformClient]:
    """
    Return a BaseWaveformClient class by built-in name or installed entry point name.

    Parameters
    ----------
    name:
        Name of the waveform client implementation to load.

    Returns
    -------
    waveform_client_class:
        The requested BaseWaveformClient subclass.

    Raises
    ------
    ConfigError
        If the waveform client implementation cannot be found or does not implement the
        BaseWaveformClient interface.

    """

    if name in _BUILTIN_WAVEFORM_CLIENTS:
        return _BUILTIN_WAVEFORM_CLIENTS[name]

    plugin_entry_points = _entry_points()

    if name not in plugin_entry_points:
        raise ConfigError(
            f"Unknown onset.name {name}. Available waveform clients: "
            f"{list_waveform_clients()}"
        )

    entry_point = plugin_entry_points[name]

    try:
        waveform_client_class = entry_point.load()
    except Exception as exception:
        raise ConfigError(
            f"Failed to load waveform client plugin {name} from "
            f"entry point {entry_point.value}."
        ) from exception

    if not isinstance(waveform_client_class, type) or not issubclass(
        waveform_client_class, BaseWaveformClient
    ):
        raise ConfigError(
            f"Waveform client plugin {name} loaded {waveform_client_class}, which is "
            "not a subclass of "
            f"{BaseWaveformClient.__module__}.{BaseWaveformClient.__name__}."
        )

    return waveform_client_class
