"""
Registry and entry-point discovery for onset function plugins.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

from quakemigrate.exceptions import ConfigError
from quakemigrate.plugins.onsets.base import Onset
from quakemigrate.plugins.onsets.stalta import STALTAOnset


ENTRY_POINT_GROUP = "quakemigrate.onsets"


_BUILTIN_ONSETS: dict[str, type[Onset]] = {
    "stalta": STALTAOnset,
}


def _entry_points() -> dict[str, EntryPoint]:
    """Return installed onset plugin entry points."""

    return {
        entry_point.name: entry_point
        for entry_point in entry_points(group=ENTRY_POINT_GROUP)
    }


def list_onsets() -> list[str]:
    """Return the names of built-in and installed onset functions."""

    return sorted([*_BUILTIN_ONSETS, *_entry_points()])


def get_onset_class(name: str) -> type[Onset]:
    """
    Return an Onset class by built-in name or installed entry point name.

    Parameters
    ----------
    name:
        Name of the onset implementation to load.

    Returns
    -------
    onset_class:
        The requested Onset subclass.

    Raises
    ------
    ConfigError
        If the onset implementation cannot be found or does not implement the Onset
        interface.

    """

    if name in _BUILTIN_ONSETS:
        return _BUILTIN_ONSETS[name]

    plugin_entry_points = _entry_points()

    if name not in plugin_entry_points:
        raise ConfigError(
            f"Unknown onset.name {name}. Available onset functions: {list_onsets()}"
        )

    entry_point = plugin_entry_points[name]

    try:
        onset_class = entry_point.load()
    except Exception as exception:
        raise ConfigError(
            f"Failed to load onset plugin {name} from entry point {entry_point.value}."
        ) from exception

    if not isinstance(onset_class, type) or not issubclass(onset_class, Onset):
        raise ConfigError(
            f"Onset plugin {name} loaded {onset_class}, which is not a subclass of "
            f"{Onset.__module__}.{Onset.__name__}."
        )

    return onset_class
