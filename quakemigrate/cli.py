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
from dataclasses import dataclass
from typing import Callable

from quakemigrate.exceptions import CLIError, ProjectError
from quakemigrate.workflow.project import require_project_root


@dataclass(frozen=True)
class RequiresProject:
    """Callable wrapper enforcing execution inside a QuakeMigrate project."""

    func: Callable[[argparse.Namespace], None]

    def __call__(self, args: argparse.Namespace) -> None:
        _ = require_project_root()
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
    (project_dir / ".qm-project").touch()  # Mark project root

    for dir_ in ["inputs", "luts", "configs/templates", "runs"]:
        (project_dir / dir_).mkdir(parents=True, exist_ok=True)

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
      configs/templates/detect.toml  -> configs/<run_name>/detect-<run_name>.toml
      configs/templates/trigger.toml -> configs/<run_name>/trigger-<run_name>.toml
      configs/templates/locate.toml  -> configs/<run_name>/locate-<run_name>.toml

    """

    project_root = require_project_root()
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

        dest = run_dir / f"{src.stem}-{args.name}{src.suffix}"
        dest.write_bytes(src.read_bytes())


def _new_lut(args: argparse.Namespace) -> None:
    """
    Create a new LUT config by copying the LUT template.

    Creates:
      luts/<lut_name>.toml

    """

    project_root = require_project_root()
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


def _build_lut(args: argparse.Namespace) -> None:
    """Construct a traveltime LUT from a .toml config file."""
    from quakemigrate.workflow.stages import lut

    lut.build_project(args.lut_name)


def _run_detect(args: argparse.Namespace) -> None:
    """Prepare and execute a Detect run from a .toml config file."""

    from quakemigrate.workflow.stages import detect

    detector = detect.prepare_project(
        args.run_name, threads=args.threads, debug=args.debug
    )

    detector.detect(args.starttime, args.endtime)


def _run_trigger(args: argparse.Namespace) -> None:
    """Prepare and execute a Trigger run from a .toml config file."""

    from quakemigrate.workflow.stages import trigger

    trigger_ = trigger.prepare_project(args.run_name, debug=args.debug)

    trigger_.trigger(args.starttime, args.endtime, region=args.region)


def _run_locate(args: argparse.Namespace) -> None:
    """Prepare and execute a Locate run from a .toml config file."""

    from quakemigrate.workflow.stages import locate

    locator = locate.prepare_project(
        args.run_name, threads=args.threads, debug=args.debug
    )

    locator.locate(args.starttime, args.endtime)


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
    p.add_argument("lut_name", type=str, help="LUT name.")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    p.set_defaults(func=RequiresProject(_build_lut))

    # Build parsers for `quakemigrate <stage>` commands
    stages = [
        ("detect", _run_detect),
        ("trigger", _run_trigger),
        ("locate", _run_locate),
    ]

    for cmd, fn in stages:
        p = sub_parser.add_parser(cmd)
        p.add_argument("run_name", type=str, help="Run name.")
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
        p.add_argument(
            "-st",
            "--starttime",
            type=str,
            help="Override for starttime.",
        )
        p.add_argument(
            "-et",
            "--endtime",
            type=str,
            help="Override for endtime.",
        )
        if cmd == "trigger":
            p.add_argument(
                "--region",
                nargs="+",
                type=float,
                help="Override for spatial filtering of triggered events."
            )
        p.set_defaults(func=RequiresProject(fn))

    args = parser.parse_args(argv)

    try:
        args.func(args)
    except CLIError as e:
        print(f"\nError:\n  {e}\n", file=sys.stderr)
        sys.exit(e.exit_code)
