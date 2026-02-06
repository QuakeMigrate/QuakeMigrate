"""
Functions for running the Locate stage of QuakeMigrate, specified at three levels:

    1. Low-level: run locate from a structured dictionary of parameters.
    2. From-file: run locate from a config file (thin layer to level 1).
    3. From-project: run locate from a project run name (thin layer to level 2).

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib
from typing import Mapping, Any, TYPE_CHECKING

from quakemigrate import QuakeScan
from quakemigrate.exceptions import ConfigError
from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.signal.pickers import GaussianPicker
from quakemigrate.workflow.config import require_key, load_toml
from quakemigrate.workflow.project import require_project_root


if TYPE_CHECKING:
    from quakemigrate.signal.onsets import Onset
    from quakemigrate.signal.pickers import Picker


def run(
    config: dict,
    run_name: str,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> None:
    """
    Run the Locate stage from an already-loaded configuration mapping.

    Parameters
    ----------
    config:
        Locate stage configuration (parsed TOML dict or similar).
    run_name:
        Name of the run (used for outputs under run_path/run_name).
    run_path:
        Base directory for run outputs (default "runs", relative to project root).
    threads:
        Optional override for scan.threads.
    debug:
        Enable debug logging.

    """

    if threads is not None and threads < 1:
        raise ConfigError("locate: threads override must be >= 1")

    station_file = pathlib.Path(require_key(config, "station_file"))
    if not station_file.exists():
        raise ConfigError(f"locate: station_file not found:\n  {station_file}")
    stations = read_stations(station_file)

    lut_file = pathlib.Path(require_key(config, "lut_file"))
    if not lut_file.exists():
        raise ConfigError(f"locate: lut_file not found:\n  {lut_file}")
    lut = read_lut(lut_file)

    archive_config = require_key(config, "archive")
    archive = Archive(
        archive_path=require_key(archive_config, "path"),
        stations=stations,
        archive_format=require_key(archive_config, "format"),
    )

    onset_config = require_key(config, "onset")
    onset = _build_onset(onset_config)

    picker_config = require_key(config, "picker")
    picker = _build_picker(picker_config, onset=onset)

    scan_config = require_key(config, "scan")
    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        picker=picker,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="debug" if debug else "info",
    )
    scan.marginal_window = require_key(scan_config, "marginal_window")
    scan.threads = (
        threads if threads is not None else require_key(scan_config, "threads")
    )
    scan.plot_event_summary = require_key(scan_config, "plot_event_summary")
    scan.write_cut_waveforms = require_key(scan_config, "write_cut_waveforms")

    scan.locate(
        require_key(scan_config, "starttime"),
        require_key(scan_config, "endtime"),
    )


def _build_onset(onset_config: dict) -> Onset:
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


def _build_picker(picker_config: dict, onset: Onset) -> Picker:
    """
    Utility for building an Picker object from config.

    Parameters
    ----------
    picker_config:
        Configuration used to build Picker object.

    Returns
    -------
    picker:
        A configured Picker object.

    Raises
    ------
    ConfigError
        If an invalid Picker type is requested.

    """

    name = require_key(picker_config, "name")

    match name:
        case "Gaussian":
            picker = GaussianPicker(onset=onset)
            picker.plot_picks = require_key(picker_config, "plot_picks")
            return picker
        case _:
            raise ConfigError(f"picker.name must be one of: ['Gaussian']")


def run_file(
    path: str | pathlib.Path,
    run_name: str,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> None:
    """
    Run the Locate stage by specifying a .toml config file.

    Parameters
    ----------
    path:
        Path to locate config file.
    run_name:
        A unique identifier for a run associated with a QuakeMigrate project.
    run_path:
        Directory to which outputs are written.
    threads:
        Optional override for scan.threads.
    debug:
        Toggle the log level to debug.

    """

    config = load_toml(pathlib.Path(path))

    run(config, run_name=run_name, run_path=run_path, threads=threads, debug=debug)


def run_project(
    run_name: str,
    project_root: str | pathlib.Path | None = None,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> None:
    """
    Run the Locate stage by a run name associated with a QuakeMigrate project.

    Parameters
    ----------
    run_name:
        A unique identifier for a run associated with a QuakeMigrate project.
    project_root:
        Override where project root is sought.
    run_path:
        Directory within project to which outputs are written.
    threads:
        Optional override for scan.threads.
    debug:
        Toggle the log level to debug.

    """

    root = require_project_root(pathlib.Path(project_root) if project_root else None)
    config_file = root / "configs" / run_name / f"locate-{run_name}.toml"

    run_file(
        config_file,
        run_name=run_name,
        run_path=root / run_path,
        threads=threads,
        debug=debug,
    )
