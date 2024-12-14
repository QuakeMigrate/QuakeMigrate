"""
Command-line interface (CLI) for the QuakeMigrate package.

This module provides the a collection of scripts for:
    - initialising a new QuakeMigrate project;
    - configuring a QuakeMigrate project;
    - and running each stage of the workflow.

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import pathlib
import shutil
import sys
import tomllib


def _init_project(args: dict) -> None:
    """
    Initialise a QuakeMigrate project directory and populate with placeholder
    configuration files.

    """

    project_dir = pathlib.Path(args.basedir) / args.name
    project_dir.mkdir(parents=True, exist_ok=True)
    for dir_ in ["inputs", "outputs/runs"]:
        (project_dir / dir_).mkdir(parents=True, exist_ok=True)

    # Create an empty file to indicate the current directory is a QM project directory
    (project_dir / ".qm-project").touch()

    # Copy a set of default config files into the new project directory
    default_config_dir = pathlib.Path(__file__).parent / "assets"
    for config_file in default_config_dir.glob("*"):
        if (project_dir / config_file.name).exists():
            continue  # Protection against overwriting a project
        shutil.copy(config_file, project_dir / config_file.name)

    if args.station_file is not None:
        station_file = pathlib.Path(args.station_file)
        if station_file.exists():
            shutil.copy(station_file, project_dir / "inputs" / station_file.name)

    if args.velocity_model is not None:
        velocity_model = pathlib.Path(args.velocity_model)
        if velocity_model.exists():
            shutil.copy(velocity_model, project_dir / "inputs" / velocity_model.name)


def _run_build_lut() -> None:
    """Construct a traveltime lookup table from a .toml config file."""

    from obspy.core import AttribDict
    from pyproj import Proj

    from quakemigrate.io import read_stations
    from quakemigrate.lut import compute_traveltimes

    with (pathlib.Path.cwd() / "lut.toml").open("rb") as f:
        parameters = tomllib.load(f)

    # --- Read in the station information file ---
    stations = read_stations(parameters["station_file"])

    # --- Define the grid specifications ---
    grid_spec = AttribDict()
    for key, value in parameters["grid_specification"].items():
        grid_spec.__setattr__(key, value)
    grid_spec.grid_proj = Proj(**parameters["grid_projection"])
    grid_spec.coord_proj = Proj(**parameters["coordinate_projection"])

    # --- Homogeneous LUT generation ---
    _ = compute_traveltimes(
        grid_spec,
        stations,
        **parameters["compute"],
        save_file=pathlib.Path.cwd() / "outputs" / f"{parameters['lut_name']}.lut",
    )


def _run_detect() -> None:
    """Prepare and execute a Detect run from a .toml config file."""

    from quakemigrate import QuakeScan
    from quakemigrate.io import Archive, read_lut, read_stations
    from quakemigrate.signal.onsets import STALTAOnset

    with (pathlib.Path.cwd() / "detect.toml").open("rb") as f:
        parameters = tomllib.load(f)

    # --- Read in station file ---
    stations = read_stations(parameters["station_file"])

    # --- Create new Archive and set path structure ---
    archive = Archive(
        archive_path=parameters["archive"]["waveform_data"],
        stations=stations,
        archive_format=parameters["archive"]["archive_format"],
    )

    # --- Load the LUT ---
    lut = read_lut(lut_file=parameters["lut_file"])

    # --- Create new Onset ---
    match parameters["onset"]["name"]:
        case "STALTA-classic":
            onset = STALTAOnset(
                position="classic",
                sampling_rate=parameters["onset"]["sampling_rate"],
            )
    onset.phases = parameters["onset"]["phases"]
    onset.bandpass_filters = parameters["onset"]["bandpass_filters"]
    onset.sta_lta_windows = parameters["onset"]["sta_lta_windows"]

    # --- Create new QuakeScan ---
    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        run_path="outputs/runs",
        run_name=parameters["run_name"],
        log=True,
        loglevel=parameters["log_level"],
    )

    # --- Set detect parameters ---
    scan.timestep = parameters["scan"]["timestep"]
    scan.threads = parameters["scan"]["threads"]

    # --- Run detect ---
    scan.detect(parameters["scan"]["starttime"], parameters["scan"]["endtime"])


def _run_trigger() -> None:
    """Create and execute a Trigger run from a .toml config file."""

    from quakemigrate import Trigger
    from quakemigrate.io import read_lut

    with (pathlib.Path.cwd() / "trigger.toml").open("rb") as f:
        parameters = tomllib.load(f)

    lut = read_lut(lut_file=parameters["lut_file"])

    # --- Create new Trigger ---
    trig = Trigger(
        lut,
        run_path="outputs/runs",
        run_name=parameters["run_name"],
        log=True,
        loglevel=parameters["log_level"],
    )

    # --- Set trigger parameters ---
    trig.marginal_window = parameters["trigger"]["marginal_window"]
    trig.min_event_interval = parameters["trigger"]["min_event_interval"]
    trig.normalise_coalescence = parameters["trigger"]["normalise_coalescence"]

    # --- Set thresholding method ---
    match parameters["threshold"]["method"]:
        case "static":
            trig.threshold_method = "static"
            trig.static_threshold = parameters["threshold"]["static_threshold"]
        case "dynamic":
            trig.threshold_method = "dynamic"
            trig.mad_window_length = parameters["threshold"]["mad_window_length"]
            trig.mad_multiplier = parameters["threshold"]["mad_multiplier"]

    # --- Run trigger ---
    trig.trigger(
        parameters["trigger"]["starttime"],
        parameters["trigger"]["endtime"],
        interactive_plot=parameters["trigger"]["interactive_plot"],
    )


def _run_locate() -> None:
    """Create and execute a Locate run from a .toml config file."""

    from quakemigrate import QuakeScan
    from quakemigrate.io import Archive, read_lut, read_stations
    from quakemigrate.signal.onsets import STALTAOnset
    from quakemigrate.signal.pickers import GaussianPicker

    with (pathlib.Path.cwd() / "locate.toml").open("rb") as f:
        parameters = tomllib.load(f)

    # --- Read in station file ---
    stations = read_stations(parameters["station_file"])

    # --- Create new Archive and set path structure ---
    archive = Archive(
        archive_path=parameters["archive"]["waveform_data"],
        stations=stations,
        archive_format=parameters["archive"]["archive_format"],
    )

    # --- Load the LUT ---
    lut = read_lut(lut_file=parameters["lut_file"])

    # --- Create new Onset ---
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
    onset.phases = parameters["onset"]["phases"]
    onset.bandpass_filters = parameters["onset"]["bandpass_filters"]
    onset.sta_lta_windows = parameters["onset"]["sta_lta_windows"]

    # --- Create new PhasePicker ---
    match parameters["picker"]["name"]:
        case "Gaussian":
            picker = GaussianPicker(onset=onset)
            picker.plot_picks = parameters["picker"]["plot_picks"]

    # --- Create new QuakeScan ---
    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        picker=picker,
        run_path="outputs/runs",
        run_name=parameters["run_name"],
        log=True,
        loglevel=parameters["log_level"],
    )

    # --- Set locate parameters ---
    scan.marginal_window = parameters["scan"]["marginal_window"]
    scan.threads = parameters["scan"]["threads"]
    scan.plot_event_summary = parameters["scan"]["plot_event_summary"]
    scan.write_cut_waveforms = parameters["scan"]["write_cut_waveforms"]

    # --- Run locate ---
    scan.locate(parameters["scan"]["starttime"], parameters["scan"]["endtime"])


STAGE_FN_MAP = {
    "lut": _run_build_lut,
    "detect": _run_detect,
    "trigger": _run_trigger,
    "locate": _run_locate,
}


def _run_stage(args):
    """Run a stage of QuakeMigrate."""

    # Check the current directory is a QM project
    if not (pathlib.Path.cwd() / ".qm-project").exists():
        print(
            "This directory is not a valid QuakeMigrate project directory.\n" "Exiting."
        )
        sys.exit(1)

    STAGE_FN_MAP[args.stage]()


FN_MAP = {
    "init": _init_project,
    "run": _run_stage,
}


def entry_point(args=None) -> None:
    """Entry point for the `quakemigrate` command-line utility."""

    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        help="Select a sub-command.",
    )

    init_parser = sub_parser.add_parser(
        "init", help="Initialise a QuakeMigrate project."
    )
    init_parser.add_argument(
        "-n",
        "--name",
        help="Specify a name for the project.",
        required=True,
    )
    init_parser.add_argument(
        "-b",
        "--basedir",
        help="Specify a root directory in which to create project directory",
        default=pathlib.Path.cwd(),
    )
    init_parser.add_argument(
        "-s",
        "--station-file",
        dest="station_file",
        help="Specify a station file to copy over to the project directory.",
    )
    init_parser.add_argument(
        "-v",
        "--velocity-model",
        dest="velocity_model",
        help="Specify a velocity model file to copy over to the project directory.",
    )

    run_parser = sub_parser.add_parser("run", help="Run a stage of QuakeMigrate.")
    run_parser.add_argument(
        "-s",
        "--stage",
        help="Specify which QuakeMigrate stage to run.",
        choices=["lut", "detect", "trigger", "locate"],
        required=True,
    )

    args = parser.parse_args(args)

    # Parse arguments and execute relevant function
    FN_MAP[args.command](args)
