"""
Command-line interface (CLI) for the QuakeMigrate package.

This module provides a collection of functions for:
    - initialising a new QuakeMigrate project;
    - configuring a QuakeMigrate project;
    - and running each stage of the workflow.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import pathlib
import shutil
import sys
import tomllib
from dataclasses import dataclass
from typing import Callable


class CLIError(Exception):
    """User-facing CLI error."""

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class ConfigError(CLIError):
    exit_code = 2


class ProjectError(CLIError):
    exit_code = 3


def _load_toml_config(
    stage: str,
    run_name: str | None = None,
    lut_name: str | None = None,
) -> dict:
    """
    Read in the TOML config file for a QuakeMigrate stage.

    Parameters
    ----------
    stage:
        QuakeMigrate stage being run (used to find default).
    run_name:
        Unique name for the QM run, used to identify configs.
    lut_name:
        Unique name for the traveltime lookup table to be used.

    Returns
    -------
    parameters:
        Stage-specific parameters for QuakeMigrate.

    """

    project_root = pathlib.Path.cwd()

    if stage in {"detect", "trigger", "locate"}:
        if not run_name:
            raise ConfigError(f"{stage}: missing run name for config resolution.")
        path = project_root / "configs" / run_name / f"{stage}-{run_name}.toml"
    elif stage == "lut":
        if not lut_name:
            raise ConfigError("lut: missing lut name for config resolution.")
        path = project_root / "luts" / f"{lut_name}.toml"
    else:
        raise ConfigError(f"Unknown stage '{stage}' for config resolution.")

    if not path.exists():
        raise ConfigError(f"{stage}: config file not found:\n  {path}")
    if not path.is_file():
        raise ConfigError(f"{stage}: config path is not a file:\n  {path}")

    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{stage}: invalid TOML in {path}:\n  {e}") from None


def _require_project_root() -> None:
    if not (pathlib.Path.cwd() / ".qm-project").exists():
        raise ProjectError(
            "This directory is not a valid QuakeMigrate project directory.\n"
            "Run from the project root (where .qm-project exists)."
        )


def _require_key(d: dict, key: str, *, ctx: str) -> object:
    try:
        return d[key]
    except KeyError:
        raise ConfigError(f"{ctx}: missing required key '{key}'") from None


@dataclass(frozen=True)
class RequiresProject:
    """Callable wrapper enforcing execution inside a QuakeMigrate project."""

    func: Callable[[argparse.Namespace], None]

    def __call__(self, args: argparse.Namespace) -> None:
        _require_project_root()
        self.func(args)


def _init_project(args: argparse.Namespace) -> None:
    """
    Initialise a QuakeMigrate project directory and populate with placeholder
    configuration files.

    Project layout:
      inputs/        - user-provided data (stations, velocity models, waveform data)
      luts/          - project-wide traveltime lookup tables
      configs/       - stage configuration files
        templates/   - template TOML files (edited to set project defaults)
        luts/        - traveltime lookup table configurations
        <run-name1>/
          detect-<run-name1>.toml
          trigger-<run-name1>.toml
          locate-<run-name1>.toml
        <run-name2>/
        ...
      runs/          - run outputs
        <run-name1>/
        <run-name2>/
        ...

    """

    project_dir = (pathlib.Path(args.basedir) / args.name).resolve()
    if (project_dir / ".qm-project").exists():
        raise ProjectError(
            f"Project already exists (found .qm-project):\n  {project_dir}"
        )
    project_dir.mkdir(parents=True, exist_ok=True)

    for dir_ in ["inputs", "luts", "configs/templates", "runs"]:
        (project_dir / dir_).mkdir(parents=True, exist_ok=True)

    # Mark project root
    (project_dir / ".qm-project").touch()

    # Copy default template config files
    assets_dir = pathlib.Path(__file__).parent / "assets"
    for config_file in assets_dir.glob("*.toml"):
        shutil.copy(config_file, project_dir / "configs/templates" / config_file.name)

    if args.station_file is not None:
        station_file = pathlib.Path(args.station_file)
        if station_file.exists():
            shutil.copy(station_file, project_dir / "inputs" / station_file.name)

    if args.velocity_model is not None:
        velocity_model = pathlib.Path(args.velocity_model)
        if velocity_model.exists():
            shutil.copy(velocity_model, project_dir / "inputs" / velocity_model.name)


def _new_run(args: argparse.Namespace) -> None:
    """
    Create a new run configuration directory by copying stage templates.

    Creates:
      configs/<run_name>/

    Copies:
      templates/detect.toml  -> configs/<run_name>/detect-<run_name>.toml
      templates/trigger.toml -> configs/<run_name>/trigger-<run_name>.toml
      templates/locate.toml  -> configs/<run_name>/locate-<run_name>.toml

    """

    project_root = pathlib.Path.cwd()
    configs_dir = project_root / "configs"
    templates_dir = configs_dir / "templates"
    run_dir = configs_dir / args.name

    if not templates_dir.exists():
        raise ProjectError(f"Missing templates directory:\n  {templates_dir}")
    if not configs_dir.exists():
        raise ProjectError(f"Missing configs directory:\n  {configs_dir}")

    if run_dir.exists():
        raise CLIError(
            f"Run config already exists:\n  {run_dir}\n"
            "Choose a different name or delete the existing directory.",
            exit_code=1,
        )

    run_dir.mkdir(parents=True, exist_ok=False)

    stage_templates = ["detect.toml", "trigger.toml", "locate.toml"]
    for fname in stage_templates:
        src = templates_dir / fname
        if not src.exists():
            raise ProjectError(f"Missing stage template:\n  {src}")

        dest = run_dir / f"{src.stem}-{args.name}.{src.suffix}"
        dest.write_bytes(src.read_bytes())


def _new_lut(args: argparse.Namespace) -> None:
    """
    Create a new LUT config by copying the LUT template.

    Creates:
      luts/<lut_name>.toml

    """

    project_root = pathlib.Path.cwd()
    templates_dir = project_root / "configs/templates"
    luts_dir = project_root / "luts"

    src = templates_dir / "lut.toml"
    if not src.exists():
        raise ProjectError(f"Missing LUT template:\n  {src}")

    luts_dir.mkdir(parents=True, exist_ok=True)

    dest = luts_dir / f"{args.name}.toml"
    if dest.exists():
        raise CLIError(f"LUT config already exists:\n  {dest}", exit_code=1)

    dest.write_bytes(src.read_bytes())


def _run_build_lut(args: argparse.Namespace) -> None:
    """Construct a traveltime lookup table from a .toml config file."""

    from obspy.core import AttribDict
    from pyproj import Proj

    from quakemigrate.io import read_stations
    from quakemigrate.lut import compute_traveltimes

    parameters = _load_toml_config(
        stage="lut",
        lut_name=args.lut_name,
    )

    stations = read_stations(parameters["station_file"])

    grid_spec = AttribDict()
    for key, value in parameters["grid_specification"].items():
        grid_spec.__setattr__(key, value)
    grid_spec.grid_proj = Proj(**parameters["grid_projection"])
    grid_spec.coord_proj = Proj(**parameters["coordinate_projection"])

    _ = compute_traveltimes(
        grid_spec,
        stations,
        **parameters["compute"],
        save_file=pathlib.Path.cwd() / "luts" / f"{args.lut_name}.lut",
    )


def _run_detect(args: argparse.Namespace) -> None:
    """Prepare and execute a Detect run from a .toml config file."""

    from quakemigrate import QuakeScan
    from quakemigrate.io import Archive, read_lut, read_stations
    from quakemigrate.signal.onsets import STALTAOnset

    parameters = _load_toml_config(
        stage="detect",
        run_name=args.run_name,
    )

    station_file = pathlib.Path(_require_key(parameters, "station_file", ctx="detect"))
    if not station_file.exists():
        raise ConfigError(f"detect: station_file not found:\n  {station_file}")

    lut_file = pathlib.Path(_require_key(parameters, "lut_file", ctx="detect"))
    if not lut_file.exists():
        raise ConfigError(f"detect: lut_file not found:\n  {lut_file}")

    archive = _require_key(parameters, "archive", ctx="detect")
    path = pathlib.Path(_require_key(archive, "path", ctx="archive"))
    if not path.exists():
        raise ConfigError(
            f"detect: archive.path not found:\n  {path}"
        )

    scan = _require_key(parameters, "scan", ctx="detect")
    if args.threads is not None and args.threads < 1:
        raise ConfigError("--threads must be >= 1")

    stations = read_stations(parameters["station_file"])

    archive = Archive(
        archive_path=parameters["archive"]["path"],
        stations=stations,
        archive_format=parameters["archive"]["format"],
    )

    lut = read_lut(lut_file=parameters["lut_file"])

    match parameters["onset"]["name"]:
        case "STALTA-classic":
            onset = STALTAOnset(
                position="classic",
                sampling_rate=parameters["onset"]["sampling_rate"],
            )
        case "STALTA-centred":
            onset = STALTAOnset(
                position="centred",
                sampling_rate=parameters["onset"]["sampling_rate"],
            )
        case _:
            raise ConfigError(
                "onset.name must be one of: ['STALTA-classic', 'STALTA-centred']"
            )
    onset.phases = parameters["onset"]["phases"]
    onset.bandpass_filters = parameters["onset"]["bandpass_filters"]
    onset.sta_lta_windows = parameters["onset"]["sta_lta_windows"]

    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        run_path="runs",
        run_name=args.run_name,
        log=True,
        loglevel="debug" if args.debug else "info",
    )
    scan.timestep = parameters["scan"]["timestep"]
    scan.threads = (
        args.threads if args.threads is not None else parameters["scan"]["threads"]
    )
    scan.detect(parameters["scan"]["starttime"], parameters["scan"]["endtime"])


def _run_trigger(args: argparse.Namespace) -> None:
    """Create and execute a Trigger run from a .toml config file."""

    from quakemigrate import Trigger
    from quakemigrate.io import read_lut

    parameters = _load_toml_config(
        stage="trigger",
        run_name=args.run_name,
    )

    lut = read_lut(lut_file=parameters["lut_file"])

    trig = Trigger(
        lut,
        run_path="runs",
        run_name=args.run_name,
        log=True,
        loglevel="debug" if args.debug else "info",
    )

    trig.marginal_window = parameters["trigger"]["marginal_window"]
    trig.min_event_interval = parameters["trigger"]["min_event_interval"]
    trig.normalise_coalescence = parameters["trigger"]["normalise_coalescence"]

    match parameters["threshold"]["method"]:
        case "static":
            trig.threshold_method = "static"
            trig.static_threshold = parameters["threshold"]["static_threshold"]
        case "mad":
            trig.threshold_method = "mad"
            trig.mad_window_length = parameters["threshold"]["mad_window_length"]
            trig.mad_multiplier = parameters["threshold"]["mad_multiplier"]
        case "median_ratio":
            trig.threshold_method = "median_ratio"
            trig.median_window_length = parameters["threshold"]["median_window_length"]
            trig.median_multiplier = parameters["threshold"]["median_multiplier"]
        case _:
            raise ConfigError(
                "threshold.method must be one of: ['static', 'mad', 'median_ratio']"
            )

    trig.trigger(
        parameters["trigger"]["starttime"],
        parameters["trigger"]["endtime"],
        interactive_plot=parameters["trigger"]["interactive_plot"],
    )


def _run_locate(args: argparse.Namespace) -> None:
    """Create and execute a Locate run from a .toml config file."""

    from quakemigrate import QuakeScan
    from quakemigrate.io import Archive, read_lut, read_stations
    from quakemigrate.signal.onsets import STALTAOnset
    from quakemigrate.signal.pickers import GaussianPicker

    parameters = _load_toml_config(
        stage="locate",
        run_name=args.run_name,
    )

    stations = read_stations(parameters["station_file"])

    archive = Archive(
        archive_path=parameters["archive"]["path"],
        stations=stations,
        archive_format=parameters["archive"]["format"],
    )

    lut = read_lut(lut_file=parameters["lut_file"])

    match parameters["onset"]["name"]:
        case "STALTA-classic":
            onset = STALTAOnset(
                position="classic",
                sampling_rate=parameters["onset"]["sampling_rate"],
            )
        case "STALTA-centred":
            onset = STALTAOnset(
                position="centred",
                sampling_rate=parameters["onset"]["sampling_rate"],
            )
        case _:
            raise ConfigError(
                "onset.name must be one of: ['STALTA-classic', 'STALTA-centred']"
            )
    onset.phases = parameters["onset"]["phases"]
    onset.bandpass_filters = parameters["onset"]["bandpass_filters"]
    onset.sta_lta_windows = parameters["onset"]["sta_lta_windows"]

    match parameters["picker"]["name"]:
        case "Gaussian":
            picker = GaussianPicker(onset=onset)
            picker.plot_picks = parameters["picker"]["plot_picks"]
        case _:
            raise ConfigError("picker.name must be on of: ['Gaussian']")

    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        picker=picker,
        run_path="runs",
        run_name=args.run_name,
        log=True,
        loglevel="debug" if args.debug else "info",
    )
    scan.marginal_window = parameters["scan"]["marginal_window"]
    scan.threads = (
        args.threads if args.threads is not None else parameters["scan"]["threads"]
    )
    scan.plot_event_summary = parameters["scan"]["plot_event_summary"]
    scan.write_cut_waveforms = parameters["scan"]["write_cut_waveforms"]
    scan.locate(parameters["scan"]["starttime"], parameters["scan"]["endtime"])


def entry_point(argv: list[str] | None = None) -> None:
    """Entry point for the `quakemigrate` command-line utility."""

    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        help="Select a sub-command.",
    )

    # Build parser for `quakemigrate init` command
    init_parser = sub_parser.add_parser(
        "init", help="Initialise a QuakeMigrate project."
    )
    init_parser.add_argument(
        "-n",
        "--name",
        help="Project name.",
        required=True,
    )
    init_parser.add_argument(
        "-b",
        "--basedir",
        help="Root directory for the project.",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
    )
    init_parser.add_argument(
        "-s",
        "--station-file",
        dest="station_file",
        help="Station file to copy into project.",
        type=pathlib.Path,
    )
    init_parser.add_argument(
        "-v",
        "--velocity-model",
        dest="velocity_model",
        help="Velocity model file to copy project.",
        type=pathlib.Path,
    )
    init_parser.set_defaults(func=_init_project)

    # Build parsers for `quakemigrate new` command
    new_parser = sub_parser.add_parser("new", help="Create new configs from templates.")
    new_sub = new_parser.add_subparsers(dest="new_what", required=True)

    new_run = new_sub.add_parser("run", help="Create a new run config set.")
    new_run.add_argument("name", help="Run name (used for configs/<name>/).")
    new_run.set_defaults(func=RequiresProject(_new_run))

    new_lut = new_sub.add_parser("lut", help="Create a new LUT config from template.")
    new_lut.add_argument("name", help="LUT name (creates luts/<name>.toml).")
    new_lut.set_defaults(func=RequiresProject(_new_lut))

    p = sub_parser.add_parser("build-lut")
    p.add_argument(
        "lut_name",
        type=str,
        help="LUT name."
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    p.set_defaults(func=RequiresProject(_run_build_lut))

    stages = [
        ("detect", _run_detect),
        ("trigger", _run_trigger),
        ("locate", _run_locate),
    ]

    for cmd, fn in stages:
        p = sub_parser.add_parser(cmd)
        p.add_argument(
            "run_name",
            type=str,
            help="Run name."
        )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug-level logging.",
        )
        p.add_argument(
            "-j",
            "--threads",
            type=int,
            help="Override number of threads for this run.",
        )
        p.set_defaults(func=RequiresProject(fn))

    args = parser.parse_args(argv)

    try:
        args.func(args)
    except CLIError as e:
        print(f"\nError:\n  {e}\n", file=sys.stderr)
        sys.exit(e.exit_code)
