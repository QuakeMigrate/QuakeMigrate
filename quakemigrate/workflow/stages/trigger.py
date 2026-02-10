"""
Functions for preparing the Trigger stage of QuakeMigrate, specified at three levels:

    1. Low-level: prepare trigger from a structured dictionary of parameters.
    2. From-file: prepare trigger from a config file (thin layer to level 1).
    3. From-project: prepare trigger from a project run name (thin layer to level 2).

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib

from quakemigrate import Trigger
from quakemigrate.exceptions import ConfigError
from quakemigrate.io import read_lut
from quakemigrate.workflow.config import require_key, load_toml
from quakemigrate.workflow.project import require_project_root


def prepare(
    config: dict,
    run_name: str,
    run_path: str = "runs",
    debug: bool = False,
) -> Trigger:
    """
    Prepare the Trigger stage from an already-loaded configuration mapping.

    Parameters
    ----------
    config:
        Trigger stage configuration (parsed TOML dict or similar).
    run_name:
        A unique identifier for the run.
    run_path:
        Directory to which outputs are written.
    debug:
        Enable debug logging.

    Returns
    -------
    trigger:
        A fully configured Trigger object.

    """

    lut_file = pathlib.Path(require_key(config, "lut_file"))
    if not lut_file.exists():
        raise ConfigError(f"trigger: lut_file not found:\n  {lut_file}")
    lut = read_lut(lut_file)

    trigger = Trigger(
        lut,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="debug" if debug else "info",
    )

    trigger_config = require_key(config, "trigger")
    trigger.marginal_window = require_key(trigger_config, "marginal_window")
    trigger.min_event_interval = require_key(trigger_config, "min_event_interval")
    trigger.normalise_coalescence = require_key(trigger_config, "normalise_coalescence")

    threshold_config = require_key(config, "threshold")
    method = require_key(threshold_config, "method")

    match method:
        case "static":
            trigger.threshold_method = "static"
            trigger.static_threshold = require_key(threshold_config, "static_threshold")
        case "mad":
            trigger.threshold_method = "mad"
            trigger.mad_window_length = require_key(
                threshold_config, "mad_window_length"
            )
            trigger.mad_multiplier = require_key(threshold_config, "mad_multiplier")
        case "median_ratio":
            trigger.threshold_method = "median_ratio"
            trigger.median_window_length = require_key(
                threshold_config, "median_window_length"
            )
            trigger.median_multiplier = require_key(
                threshold_config, "median_multiplier"
            )
        case _:
            raise ConfigError(
                "trigger.threshold.method must be one of: "
                "['static', 'mad', 'median_ratio']"
            )

    return trigger


def prepare_file(
    path: str | pathlib.Path,
    run_name: str,
    run_path: str = "runs",
    debug: bool = False,
    basepath: pathlib.Path | None = None,
) -> Trigger:
    """
    Prepare the Trigger stage by specifying a .toml config file.

    Parameters
    ----------
    path:
        Path to trigger config file.
    run_name:
        A unique identifier for the run.
    run_path:
        Directory to which outputs are written.
    debug:
        Toggle the log level to debug.
    basepath:
        Optionally specify a root directory to resolve relative paths against.

    Returns
    -------
    trigger:
        A fully configured Trigger object.

    """

    config = load_toml(pathlib.Path(path), basepath=basepath)

    trigger = prepare(config, run_name=run_name, run_path=run_path, debug=debug)

    return trigger


def prepare_project(
    run_name: str,
    project_root: str | pathlib.Path | None = None,
    run_path: str = "runs",
    debug: bool = False,
) -> Trigger:
    """
    Prepare the Trigger stage by a run name associated with a QuakeMigrate project.

    Parameters
    ----------
    run_name:
        A unique identifier for a run associated with a QuakeMigrate project.
    project_root:
        Override where project root is sought.
    run_path:
        Directory within project to which outputs are written.
    debug:
        Toggle the log level to debug.

    Returns
    -------
    trigger:
        A fully configured Trigger object.

    """

    root = require_project_root(pathlib.Path(project_root) if project_root else None)
    config_file = root / "configs" / run_name / f"trigger-{run_name}.toml"

    trigger = prepare_file(
        config_file,
        run_name=run_name,
        run_path=root / run_path,
        debug=debug,
        basepath=root,
    )

    return trigger
