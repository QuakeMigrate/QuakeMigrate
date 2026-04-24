"""
Functions for running the LUT building stage of QuakeMigrate, specified at three levels:

    1. Low-level: run build-lut from a structured dictionary of parameters.
    2. From-file: run build-lut from a config file (thin layer to level 1).
    3. From-project: run build-lut from a project run name (thin layer to level 2).

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib

from obspy.core import AttribDict
from pyproj import Proj

from quakemigrate.io import read_stations, read_vmodel
from quakemigrate.lut import compute_traveltimes
from quakemigrate.workflow.config import load_toml
from quakemigrate.workflow.project import require_project_root


def build(config: dict, save_to: pathlib.Path) -> None:
    """
    Build a LUT from an already-loaded configuration mapping.

    Parameters
    ----------
    config:
        Build LUT stage configuration (parsed TOML dict or similar).
    save_to:
        Path to which to save the LUT file.

    """

    stations = read_stations(config["station_file"])

    grid_spec = AttribDict()
    for key, value in config["grid_specification"].items():
        grid_spec.__setattr__(key, value)
    grid_spec.grid_proj = Proj(**config["grid_projection"])
    grid_spec.coord_proj = Proj(**config["coordinate_projection"])

    if "vmodel_file" in config.keys():
        config["compute"]["vmod"] = read_vmodel(config["vmodel_file"])

    _ = compute_traveltimes(
        grid_spec,
        stations,
        **config["compute"],
        save_file=save_to,
    )


def build_file(path: str | pathlib.Path, save_to: pathlib.Path) -> None:
    """
    Build a LUT by specifying a .toml config file.

    Parameters
    ----------
    path:
        Path to LUT config file.
    save_to:
        Path to which to save the LUT file.

    """

    config = load_toml(pathlib.Path(path))

    build(config, save_to=save_to)


def build_project(
    lut_name: str,
    project_root: str | pathlib.Path | None = None,
) -> None:
    """
    Build a LUT by specifying a LUT name associated with a QuakeMigrate project.

    Parameters
    ----------
    lut_name:
        A unique identifier for the LUT.
    project_root:
        Override where project root is sought.

    """

    root = require_project_root(pathlib.Path(project_root) if project_root else None)
    config_file = root / "luts" / f"{lut_name}.toml"

    build_file(config_file, save_to=root / "luts" / f"{lut_name}.lut")
