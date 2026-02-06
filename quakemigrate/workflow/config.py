"""
Utilities for working with QuakeMigrate config files.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

from quakemigrate.exceptions import ConfigError


def load_toml(path: pathlib.Path) -> dict:
    """
    Read TOML configuration file.

    Parameters
    ----------
    path:
        Path to TOML configuration file.

    Returns
    -------
    config:
        Configuration file parsed into dict structure.

    Raises
    ------
    ConfigError
        If the file does not exist or is not a valid format.

    """

    if not path.exists():
        raise ConfigError(f"config file not found:\n  {path}")
    if not path.is_file():
        raise ConfigError(f"config path is not a file:\n  {path}")

    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}:\n  {e}") from None


def require_key(d: dict, key: str) -> Any:
    """
    Access config value by key.

    Parameters
    ----------
    d:
        Config in dict structure.
    key:
        Parameter to be accessed.

    Returns
    -------
    value:
        Value of parameter requested.

    Raises
    ------
    ConfigError
        If key is missing.

    """

    try:
        return d[key]
    except KeyError:
        raise ConfigError(f"missing required key '{key}'") from None
