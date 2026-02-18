"""
Common utilities for building components for workflow stages.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from obspy.core import AttribDict

from quakemigrate.exceptions import ConfigError
from quakemigrate.io import Archive, read_response_inv
from quakemigrate.plugins.magnitudes import LocalMag
from quakemigrate.plugins.onsets import STALTAOnset
from quakemigrate.plugins.pickers import GaussianPicker
from quakemigrate.workflow.config import get_required_key, pop_required_key


if TYPE_CHECKING:
    import pandas as pd

    from quakemigrate.plugins.onsets import Onset
    from quakemigrate.plugins.pickers import PhasePicker


def build_archive(archive_config: dict, stations: pd.DataFrame) -> Archive:
    """
    Utility for building an Archive object from config.

    Parameters
    ----------
    archive_config:
        Configuration used to build Archive object.

    Returns
    -------
    archive:
        A configured Archive object.

    """

    try:
        response_inv = read_response_inv(archive_config.pop("response_inv"))
    except (IsADirectoryError, FileNotFoundError, TypeError, KeyError):
        response_inv = None
    response_removal_params = archive_config.pop("response_removal_params", {})

    archive = Archive(
        archive_path=pop_required_key(archive_config, "path"),
        stations=stations,
        archive_format=pop_required_key(archive_config, "format"),
        response_inv=response_inv,
        response_removal_params=response_removal_params,
        **archive_config,
    )

    return archive


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

    name = pop_required_key(onset_config, "name")

    match name:
        case "STALTA-classic":
            onset = STALTAOnset(
                position="classic",
                sampling_rate=pop_required_key(onset_config, "sampling_rate"),
                **onset_config,
            )
        case "STALTA-centred":
            onset = STALTAOnset(
                position="centred",
                sampling_rate=pop_required_key(onset_config, "sampling_rate"),
                **onset_config,
            )
        case _:
            raise ConfigError(
                f"onset.name must be one of: ['STALTA-classic', 'STALTA-centred']"
            )

    onset.phases = get_required_key(onset_config, "phases")
    onset.bandpass_filters = get_required_key(onset_config, "bandpass_filters")
    onset.sta_lta_windows = get_required_key(onset_config, "sta_lta_windows")

    return onset


def build_picker(picker_config: dict, onset: Onset, **_: Any) -> PhasePicker:
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

    name = pop_required_key(picker_config, "name")

    match name:
        case "Gaussian":
            picker = GaussianPicker(onset=onset, **picker_config)
        case _:
            raise ConfigError(f"picker.name must be one of: ['Gaussian']")

    return picker


def build_magnitudes(config: dict, **_: Any) -> LocalMag:
    """
    Build a LocalMag magnitude calculator from config.

    Parameters
    ----------
    config:
        Mapping containing magnitude plugin configuration.

    Returns
    -------
    mags:
        Configured LocalMag instance.

    Raises
    ------
    ConfigError
        If required subkeys are missing or have invalid types.

    """

    plot_amplitudes = config.get("plot_amplitudes", True)

    amp_config = pop_required_key(config, "amp")
    amp_params = AttribDict()
    for k, v in amp_config.items():
        amp_params[k] = v

    mag_config = pop_required_key(config, "mag")
    mag_params = AttribDict()
    for k, v in mag_config.items():
        mag_params[k] = v

    mags = LocalMag(
        amp_params=amp_params,
        mag_params=mag_params,
        plot_amplitudes=plot_amplitudes,
    )

    return mags
