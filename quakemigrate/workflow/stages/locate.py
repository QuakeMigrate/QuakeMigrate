"""
Functions for preparing the Locate stage of QuakeMigrate, specified at three levels:

    1. Low-level: prepare locate from a structured dictionary of parameters.
    2. From-file: prepare locate from a config file (thin layer to level 1).
    3. From-project: prepare locate from a project run name (thin layer to level 2).

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib

from quakemigrate import QuakeScan
from quakemigrate.exceptions import ConfigError
from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.workflow.builders import build_onset, build_picker
from quakemigrate.workflow.config import require_key, load_toml
from quakemigrate.workflow.project import require_project_root


def prepare(
    config: dict,
    run_name: str,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> QuakeScan:
    """
    Prepare a reusable QuakeScan object for Locate.

    Parameters
    ----------
    config:
        Locate stage configuration (parsed TOML dict or similar).
    run_name:
        A unique identifier for the run.
    run_path:
        Directory to which outputs are written.
    threads:
        Optional override for QuakeScan.threads.
    debug:
        Enable debug logging.

    Returns
    -------
    locator:
        A fully configured QuakeScan object.

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
    onset = build_onset(onset_config)

    picker_config = require_key(config, "picker")
    picker = build_picker(picker_config, onset=onset)

    scan_config = require_key(config, "scan")
    locator = QuakeScan(
        archive,
        lut,
        onset=onset,
        picker=picker,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="debug" if debug else "info",
    )
    locator.marginal_window = require_key(scan_config, "marginal_window")
    locator.threads = (
        threads if threads is not None else require_key(scan_config, "threads")
    )
    locator.plot_event_summary = require_key(scan_config, "plot_event_summary")
    locator.write_cut_waveforms = require_key(scan_config, "write_cut_waveforms")

    return locator


def prepare_file(
    path: str | pathlib.Path,
    run_name: str,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
    basepath: pathlib.Path | None = None,
) -> QuakeScan:
    """
    Prepare the Locate stage by specifying a .toml config file.

    Parameters
    ----------
    path:
        Path to locate config file.
    run_name:
        A unique identifier for the run.
    run_path:
        Directory to which outputs are written.
    threads:
        Optional override for QuakeScan.threads.
    debug:
        Toggle the log level to debug.
    basepath:
        Optionally specify a root directory to resolve relative paths against.

    Returns
    -------
    locator:
        A fully configured QuakeScan object.

    """

    config = load_toml(pathlib.Path(path), basepath=basepath)

    locator = prepare(
        config, run_name=run_name, run_path=run_path, threads=threads, debug=debug
    )

    return locator


def prepare_project(
    run_name: str,
    project_root: str | pathlib.Path | None = None,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> QuakeScan:
    """
    Prepare the Locate stage by a run name associated with a QuakeMigrate project.

    Parameters
    ----------
    run_name:
        A unique identifier for a run associated with a QuakeMigrate project.
    project_root:
        Override where project root is sought.
    run_path:
        Directory within project to which outputs are written.
    threads:
        Optional override for QuakeScan.threads.
    debug:
        Toggle the log level to debug.

    Returns
    -------
    locator:
        A fully configured QuakeScan object.

    """

    root = require_project_root(pathlib.Path(project_root) if project_root else None)
    config_file = root / "configs" / run_name / f"locate-{run_name}.toml"

    locator = prepare_file(
        config_file,
        run_name=run_name,
        run_path=root / run_path,
        threads=threads,
        debug=debug,
        basepath=root,
    )

    return locator
