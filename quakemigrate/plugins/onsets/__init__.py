"""
The :mod:`quakemigrate.plugins.onsets` module handles the generation of Onset functions.
The default method uses the ratio between the short-term and long-term averages of the
signal amplitude.

Feel free to contribute more Onset function options!

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from typing import Any, Mapping

from quakemigrate.plugins.onsets.base import Onset, OnsetData
from quakemigrate.plugins.onsets.registry import get_onset_class


def make_onset_function(config: Mapping[str, Any]) -> Onset:
    """
    Utility for building an Onset object from config.

    Parameters
    ----------
    config:
        Configuration used to build Onset object.

    Returns
    -------
    onset:
        A configured Onset object.

    """

    onset_config = dict(config)

    onset_name = onset_config.pop("name")
    sampling_rate = onset_config.pop("sampling_rate")

    onset_class = get_onset_class(onset_name)

    return onset_class(
        sampling_rate=sampling_rate,
        **onset_config,
    )


__all__ = ["Onset", "OnsetData", "make_onset_function"]
