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

from quakemigrate.exceptions import ConfigError, ProjectError


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


def stage_config_path(
    *,
    stage: str,
    run_name: str | None = None,
    lut_name: str | None = None,
    project_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Resolve config paths using the existing project layout.

    - detect/trigger/locate: configs/<run_name>/<stage>-<run_name>.toml
    - lut config:            luts/<lut_name>.toml
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
