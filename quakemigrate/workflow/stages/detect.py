"""
Functions for preparing the Detect stage of QuakeMigrate, specified at three levels:

    1. Low-level: prepare detect from a structured dictionary of parameters.
    2. From-file: prepare detect from a config file (thin layer to level 1).
    3. From-project: prepare detect from a project run name (thin layer to level 2).

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
from quakemigrate.io import read_lut, read_stations
from quakemigrate.workflow.builders import build_archive, build_onset
from quakemigrate.workflow.config import get_required_key, load_toml, pop_required_key
from quakemigrate.workflow.project import require_project_root


def prepare(
    config: dict,
    run_name: str,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> QuakeScan:
    """
    Prepare a reusable QuakeScan object for Detect.

    Parameters
    ----------
    config:
        Detect stage configuration (parsed TOML dict or similar).
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
    detector:
        A fully configured QuakeScan object.

    """

    if threads is not None and threads < 1:
        raise ConfigError("detect: threads override must be >= 1")

    station_file = pathlib.Path(get_required_key(config, "station_file"))
    if not station_file.exists():
        raise ConfigError(f"detect: station_file not found:\n  {station_file}")
    stations = read_stations(station_file)

    lut_config = pop_required_key(config, "lut")
    lut_file = pathlib.Path(get_required_key(lut_config, "file"))
    if not lut_file.exists():
        raise ConfigError(f"detect: lut.file not found:\n  {lut_file}")
    lut = read_lut(lut_file)
    lut.decimate(lut_config.get("decimation"), inplace=True)

    archive_config = pop_required_key(config, "archive")
    archive = build_archive(archive_config, stations=stations)

    onset_config = pop_required_key(config, "onset")
    onset = build_onset(onset_config)

    scan_config = pop_required_key(config, "scan")
    detector = QuakeScan(
        archive,
        lut,
        onset=onset,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="debug" if debug else "info",
        **scan_config,
    )
    detector.timestep = get_required_key(scan_config, "timestep")
    detector.threads = (
        threads if threads is not None else get_required_key(scan_config, "threads")
    )

    return detector


def prepare_file(
    path: str | pathlib.Path,
    run_name: str,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
    basepath: pathlib.Path | None = None,
) -> QuakeScan:
    """
    Prepare the Detect stage by specifying a .toml config file.

    Parameters
    ----------
    path:
        Path to detect config file.
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
    detector:
        A fully configured QuakeScan object.

    """

    config = load_toml(pathlib.Path(path), basepath=basepath)

    detector = prepare(
        config, run_name=run_name, run_path=run_path, threads=threads, debug=debug
    )

    return detector


def prepare_project(
    run_name: str,
    project_root: str | pathlib.Path | None = None,
    run_path: str = "runs",
    threads: int | None = None,
    debug: bool = False,
) -> QuakeScan:
    """
    Prepare the Detect stage by a run name associated with a QuakeMigrate project.

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
    detector:
        A fully configured QuakeScan object.

    """

    root = require_project_root(pathlib.Path(project_root) if project_root else None)
    config_file = root / "configs" / run_name / f"detect-{run_name}.toml"

    detector = prepare_file(
        config_file,
        run_name=run_name,
        run_path=root / run_path,
        threads=threads,
        debug=debug,
        basepath=root,
    )

    return detector
