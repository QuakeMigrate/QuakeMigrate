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


def load_toml(config_file: pathlib.Path, basepath: pathlib.Path | None = None) -> dict:
    """
    Read TOML configuration file.

    Parameters
    ----------
    config_file:
        Path to TOML configuration file.
    basepath:
        Optionally specify a root directory to resolve relative paths against.

    Returns
    -------
    config:
        Configuration file parsed into dict structure.

    Raises
    ------
    ConfigError
        If the file does not exist or is not a valid format.

    """

    if not config_file.exists():
        raise ConfigError(f"config file not found:\n  {config_file}")
    if not config_file.is_file():
        raise ConfigError(f"config path is not a file:\n  {config_file}")

    try:
        with config_file.open("rb") as f:
            config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {config_file}:\n  {e}") from None

    if basepath is not None:
        config = resolve_config_paths(config, basepath)

    return config


_PATH_KEYS = {"station_file", "lut_file", "path"}


def resolve_config_paths(
    config: dict,
    basepath: pathlib.Path,
) -> dict:
    """
    Resolve relative paths in config file against some base directory.

    Parameters
    ----------
    config:
        Configuration file parsed into dict structure.
    basepath:
        Root directory to resolve relative paths against.

    Returns
    -------
    config:
        The config object with resolved paths.

    """

    def recurse(obj, key=None):
        if isinstance(obj, dict):
            return {k: recurse(v, k) for k, v in obj.items()}

        if isinstance(obj, list):
            return [recurse(v, key) for v in obj]

        if isinstance(obj, str) and key is not None:
            if key in _PATH_KEYS:
                p = pathlib.Path(obj).expanduser()
                if not p.is_absolute():
                    p = (basepath / p).resolve()
                return str(p)

        return obj

    return recurse(config)


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
