"""
Common utilities for building components for workflow stages.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quakemigrate.exceptions import ConfigError
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.signal.pickers import GaussianPicker
from quakemigrate.workflow.config import require_key


if TYPE_CHECKING:
    from quakemigrate.signal.onsets import Onset
    from quakemigrate.signal.pickers import PhasePicker


def build_onset(onset_config: dict) -> Onset:
    """
    Utility for building an Onset object from config.

    Parameters
    ----------
    onset_config:
        Configuration used to build Onset object.

    Returns
    -------
    onset:
        A configured Onset object.

    Raises
    ------
    ConfigError
        If an invalid Onset type is requested.

    """

    name = require_key(onset_config, "name")

    match name:
        case "STALTA-classic":
            onset = STALTAOnset(
                position="classic",
                sampling_rate=require_key(onset_config, "sampling_rate"),
            )
        case "STALTA-centred":
            onset = STALTAOnset(
                position="centred",
                sampling_rate=require_key(onset_config, "sampling_rate"),
            )
        case _:
            raise ConfigError(
                f"onset.name must be one of: ['STALTA-classic', 'STALTA-centred']"
            )

    onset.phases = require_key(onset_config, "phases")
    onset.bandpass_filters = require_key(onset_config, "bandpass_filters")
    onset.sta_lta_windows = require_key(onset_config, "sta_lta_windows")

    return onset


def build_picker(picker_config: dict, onset: Onset) -> PhasePicker:
    """
    Utility for building an PhasePicker object from config.

    Parameters
    ----------
    picker_config:
        Configuration used to build PhasePicker object.
    onset:
        Onset function used for picking.

    Returns
    -------
    picker:
        A configured PhasePicker object.

    Raises
    ------
    ConfigError
        If an invalid PhasePicker type is requested.

    """

    name = require_key(picker_config, "name")

    match name:
        case "Gaussian":
            picker = GaussianPicker(onset=onset)
            picker.plot_picks = require_key(picker_config, "plot_picks")
        case _:
            raise ConfigError(f"picker.name must be one of: ['Gaussian']")

    return picker
