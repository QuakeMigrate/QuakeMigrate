"""
QuakeMigrate project management utilities.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib
import shutil
from importlib.resources import files
from typing import Literal

from quakemigrate.exceptions import ConfigError, ProjectError


def init_project(
    basedir: str | pathlib.Path,
    name: str,
) -> pathlib.Path:
    """
    Initialise a QuakeMigrate project directory and populate with placeholder
    configuration files.

    Project layout:
      inputs/        - user-provided data (stations, velocity models, waveform data)
      luts/          - project-wide traveltime lookup tables
      configs/       - stage configuration files
        templates/   - template TOML files (edited to set project defaults)
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

    Parameters
    ----------
    basedir:
        Root directory in which to create QuakeMigrate project.
    name:
        Name of QuakeMigrate project.

    Returns
    -------
    project_dir:
        Resolved project directory path.

    """

    project_dir = (pathlib.Path(basedir) / name).resolve()
    if (project_dir / ".qm-project").exists():
        raise ProjectError(
            f"Project already exists (found .qm-project):\n  {project_dir}"
        )
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".qm-project").touch()  # Mark project root

    for dir_ in ["inputs", "luts", "configs/templates", "runs"]:
        (project_dir / dir_).mkdir(parents=True, exist_ok=True)

    # Copy default template config files
    assets_dir = files("quakemigrate") / "assets"
    for config_file in assets_dir.glob("*.toml"):
        shutil.copy(config_file, project_dir / "configs/templates" / config_file.name)

    return project_dir


def require_project_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """
    Validate the current directory is either the root, or a subdirectory of the root,
    of a QuakeMigrate project by searching upwards for `.qm-project`.

    Parameters
    ----------
    start:
        The starting directory for the search.

    Returns
    -------
    parent:
        The project root directory.

    Raises
    ------
    ProjectError
        If no `.qm-project` marker file is found in any parent directory.

    """

    path = (start or pathlib.Path.cwd()).resolve()

    for parent in [path, *path.parents]:
        marker = parent / ".qm-project"
        if marker.exists():
            return parent

    raise ProjectError(
        "This directory is not inside a valid QuakeMigrate project.\n"
        "No `.qm-project` marker file was found in this directory or any parent.\n"
        "Run `quakemigrate init` to create a new project."
    )


Stage = Literal["detect", "trigger", "locate", "lut"]


def stage_config_path(
    *,
    stage: Stage,
    run_name: str | None = None,
    lut_name: str | None = None,
    project_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Resolve config paths using the existing project layout.

    - detect/trigger/locate: configs/<run_name>/<stage>-<run_name>.toml
    - lut config:            luts/<lut_name>.toml

    Parameters
    ----------
    stage:
        Name of the processing stage whose configuration path should be resolved.
    run_name:
        Name of the run. Required for detect, trigger, and locate stages.
    lut_name:
        Name of the lookup-table configuration. Required for the lut stage.
    project_root:
        Optional project root directory. If omitted, the project root is resolved using
        :func:`require_project_root`.

    Returns
    -------
    pathlib.Path
        Absolute or project-root-relative path to the stage configuration file,
        depending on the value returned by :func:`require_project_root`.

    Raises
    ------
    ConfigError
        Raised if the stage is unknown, or if the required run or LUT name is missing.

    """

    root = require_project_root(project_root)

    if stage in {"detect", "trigger", "locate"}:
        if not run_name:
            raise ConfigError(f"{stage}: missing run name for config resolution.")
        return root / "configs" / run_name / f"{stage}-{run_name}.toml"

    if stage == "lut":
        if not lut_name:
            raise ConfigError("lut: missing lut name for config resolution.")
        return root / "luts" / f"{lut_name}.toml"

    raise ConfigError(f"Unknown stage '{stage}' for config resolution.")
