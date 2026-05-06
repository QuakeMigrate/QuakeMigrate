"""
Collection of plugins for use with the core QuakeMigrate package.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Mapping, Protocol

from quakemigrate.exceptions import ConfigError


BUILTIN_PLUGINS = {
    "EventSummary2D": "quakemigrate.plugins.visualisation:EventSummary2DPlugin",
    "EventSummary3D": "quakemigrate.plugins.visualisation:EventSummary3DPlugin",
    "GaussianPicker": "quakemigrate.plugins.pickers:build_gaussian_picker",
}


class Plugin(Protocol):
    stage: str
    order: int
    name: str

    def run(self, **ctx: Any) -> Mapping[str, Any] | None:
        """Execute plugin."""
        ...


def call_by_signature(fn: Callable, available: dict[str, Any]):
    """
    Utility function that handles dependency injection based on the signature of the
    function.

    """

    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in available:
            kwargs[name] = available[name]
        elif param.default is inspect._empty:
            raise TypeError(f"Missing required dependency '{name}' for {fn}")
    return fn(**kwargs)


def import_from_string(path: str) -> Any:
    """
    Import an object from a "module:object" string.

    Parameters
    ----------
    path:
        Import path in the form "package.module:object_name".

    Returns
    -------
    obj:
        Imported Python object.

    Raises
    ------
    ConfigError
        If the import string is malformed, the module cannot be imported, or
        the named object does not exist within the module.

    """

    if ":" not in path:
        raise ConfigError(
            f"Invalid plugin path {path}. Expected format 'package.module:ObjectName'."
        )

    module_name, object_name = [s.strip() for s in path.split(":")]

    if not module_name or not object_name:
        raise ConfigError(
            f"Invalid plugin path {path}. Expected format 'package.module:ObjectName'."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ConfigError(
            f"Could not import plugin module {module_name} from {path}."
        ) from e

    try:
        return getattr(module, object_name)
    except AttributeError as e:
        raise ConfigError(f"Module {module_name} does not define {object_name}.") from e


def construct_plugin(
    plugin_config: Mapping[str, Any],
    **context: Any,
) -> Any:
    """
    Construct a plugin object from a plugin specification.

    The specification must contain either:
    - plugin: a fully qualified import path of the form "package.module:ObjectName", or
    - name: a short built-in alias resolved through builtins.

    Any remaining keys in ``plugin_config`` are treated as configuration arguments and
    are injected into the target constructor or factory function by signature.

    Parameters
    ----------
    plugin_config:
        Plugin specification dictionary.
    **context:
        Additional dependencies made available for signature-based injection.

    Returns
    -------
    plugin:
        Constructed plugin object.

    Raises
    ------
    ConfigError
        If the specification is invalid, the built-in alias is unknown, or the
        target cannot be imported.

    """

    plugin_config = dict(plugin_config)

    plugin_path = plugin_config.pop("plugin", None)
    plugin_name = plugin_config.pop("name", None)

    if plugin_path and plugin_name:
        raise ConfigError("Plugin config must contain only one of 'plugin' or 'name'.")

    if plugin_path:
        target = import_from_string(plugin_path)
    elif plugin_name:
        try:
            target = import_from_string(BUILTIN_PLUGINS[plugin_name])
        except KeyError as e:
            allowed = ", ".join(x for x in sorted(BUILTIN_PLUGINS))
            raise ConfigError(
                f"Unknown built-in plugin {plugin_name}. Expected one of: [{allowed}]"
            ) from e
    else:
        raise ConfigError(
            "Plugin specification must contain either 'plugin' or 'name'."
        )

    return call_by_signature(target, {**plugin_config, **context})
