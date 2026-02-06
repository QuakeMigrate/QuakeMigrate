"""
Functions for running the Trigger stage of QuakeMigrate, specified at three levels:

    1. Low-level: run trigger from a structured dictionary of parameters.
    2. From-file: run trigger from a config file (thin layer to level 1).
    3. From-project: run trigger from a project run name (thin layer to level 2).

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


def run(
    config: dict,
    run_name: str,
    run_path: str = "runs",
    debug: bool = False,
) -> None:
    """
    Run the Trigger stage from an already-loaded configuration mapping.

    Parameters
    ----------
    config:
        Trigger stage configuration (parsed TOML dict or similar).
    run_name:
        Name of the run (used for outputs under run_path/run_name).
    run_path:
        Base directory for run outputs (default "runs", relative to project root if used).
    debug:
        Enable debug logging.

    """

    lut_file = pathlib.Path(require_key(config, "lut_file"))
    if not lut_file.exists():
        raise ConfigError(f"detect: lut_file not found:\n  {lut_file}")
    lut = read_lut(lut_file)

    trig = Trigger(
        lut,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="debug" if debug else "info",
    )

    trigger_cfg = require_key(config, "trigger")
    trig.marginal_window = require_key(trigger_cfg, "marginal_window")
    trig.min_event_interval = require_key(trigger_cfg, "min_event_interval")
    trig.normalise_coalescence = require_key(trigger_cfg, "normalise_coalescence")

    threshold_cfg = require_key(config, "threshold")
    method = require_key(threshold_cfg, "method")

    match method:
        case "static":
            trig.threshold_method = "static"
            trig.static_threshold = require_key(threshold_cfg, "static_threshold")
        case "mad":
            trig.threshold_method = "mad"
            trig.mad_window_length = require_key(threshold_cfg, "mad_window_length")
            trig.mad_multiplier = require_key(threshold_cfg, "mad_multiplier")
        case "median_ratio":
            trig.threshold_method = "median_ratio"
            trig.median_window_length = require_key(
                threshold_cfg, "median_window_length"
            )
            trig.median_multiplier = require_key(threshold_cfg, "median_multiplier")
        case _:
            raise ConfigError(
                "trigger.threshold.method must be one of: ['static', 'mad', 'median_ratio']"
            )

    trig.trigger(
        require_key(trigger_cfg, "starttime"),
        require_key(trigger_cfg, "endtime"),
        interactive_plot=require_key(trigger_cfg, "interactive_plot"),
    )


def run_file(
    path: str | pathlib.Path,
    run_name: str,
    run_path: str = "runs",
    debug: bool = False,
) -> None:
    """
    Run the Trigger stage by specifying a .toml config file.

    Parameters
    ----------
    path:
        Path to trigger config file.
    run_name:
        A unique identifier for a run associated with a QuakeMigrate project.
    run_path:
        Directory to which outputs are written.
    debug:
        Toggle the log level to debug.

    """

    config = load_toml(pathlib.Path(path))

    run(config, run_name=run_name, run_path=run_path, debug=debug)


def run_project(
    run_name: str,
    project_root: str | pathlib.Path | None = None,
    run_path: str = "runs",
    debug: bool = False,
) -> None:
    """
    Run the Trigger stage by a run name associated with a QuakeMigrate project.

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

    """

    root = require_project_root(pathlib.Path(project_root) if project_root else None)
    config_file = root / "configs" / run_name / f"trigger-{run_name}.toml"

    run_file(config_file, run_name=run_name, run_path=root / run_path, debug=debug)
