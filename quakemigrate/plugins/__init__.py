"""
Collection of plugins for use with the core QuakeMigrate package.

Plugins are resolved by name from either QuakeMigrate's built-in plugin registry or from
installed third-party packages that expose entry points in the ``quakemigrate.plugins``
group.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import importlib
import inspect
from importlib.metadata import entry_points
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from quakemigrate.exceptions import ConfigError


BUILTIN_PLUGINS = {
    "EventSummary2D": "quakemigrate.plugins.visualisation:EventSummary2DPlugin",
    "EventSummary3D": "quakemigrate.plugins.visualisation:EventSummary3DPlugin",
    "GaussianPicker": "quakemigrate.plugins.pickers:build_gaussian_picker",
    "LocalMagnitude": "quakemigrate.plugins.magnitudes:build_local_magnitude_plugin",
}

ENTRY_POINT_GROUP = "quakemigrate.plugins"


@runtime_checkable
class Plugin(Protocol):
    """
    Protocol for QuakeMigrate plugins.

    Plugins may be implemented as any object that provides the required metadata
    attributes and a ``run`` method. They do not need to inherit from a common base
    class.

    Attributes
    ----------
    stage:
        Name of the processing stage in which the plugin should run.
    order:
        Relative ordering of the plugin within its stage. Lower values run earlier.
    name:
        Human-readable plugin name.

    """

    stage: str
    order: int
    name: str

    def run(self, **ctx: Any) -> Mapping[str, Any] | None:
        """
        Execute the plugin.

        Parameters
        ----------
        **ctx:
            Runtime context made available to the plugin.

        Returns
        -------
        updates:
            Optional mapping of context updates produced by the plugin.

        """
        ...


def call_by_signature(fn: Callable, available: dict[str, Any]):
    """
    Call a constructor or factory function using signature-based dependency injection.

    Parameters
    ----------
    fn:
        Callable to invoke.
    available:
        Mapping of available configuration values and runtime dependencies.

    Returns
    -------
    result:
        Object returned by ``fn``.

    Raises
    ------
    ConfigError
        If the callable's signature cannot be inspected, if a required argument is
        missing, or if the callable cannot be invoked with the selected arguments.

    """

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"could not inspect signature for plugin target {fn}.") from e

    kwargs = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in available:
            kwargs[name] = available[name]
        elif parameter.default is inspect.Parameter.empty:
            raise ConfigError(f"missing required dependency '{name}' for {fn}")

    try:
        return fn(**kwargs)
    except TypeError as e:
        raise ConfigError(f"could not construct plugin from {fn}: {e}") from e


def load_builtin_plugin(name: str, import_path: str) -> Any:
    """
    Load a built-in plugin target from an internal module:object import path.

    Parameters
    ----------
    name:
        Built-in plugin name.
    import_path:
        Import path in the form "package.module:ObjectName".

    Returns
    -------
    target:
        Imported plugin class or factory function.

    Raises
    ------
    ConfigError
        If the import path is malformed, the module cannot be imported, or the object is
        not defined by the module.

    """

    try:
        module_name, object_name = import_path.split(":", 1)
    except ValueError as e:
        raise ConfigError(
            f"invalid built-in plugin path {import_path} for {name}."
        ) from e

    try:
        module = importlib.import_module(module_name)
        return getattr(module, object_name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(
            f"could not load built-in plugin {name} from {import_path}."
        ) from e


def get_entry_point_plugins() -> dict[str, Any]:
    """
    Return plugin targets registered by installed third-party packages.

    Third-party packages should expose plugins using the quakemigrate.plugins
    entry-point group. For example, in pyproject.toml::

        [project.entry-points."quakemigrate.plugins"]
        MyPlugin = "my_package.plugins:MyPlugin"

    Returns
    -------
    plugins:
        Mapping from entry-point name to loaded plugin class or factory function.

    Raises
    ------
    ConfigError
        If an entry point cannot be loaded.

    """

    plugins = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            plugins[entry_point.name] = entry_point.load()
        except Exception as e:
            raise ConfigError(
                f"Could not load plugin entry point {entry_point.name} "
                f"from {entry_point.value}."
            ) from e

    return plugins


def get_available_plugins() -> dict[str, Any]:
    """
    Return all available plugin targets.

    Available plugins include QuakeMigrate's built-in plugins and plugins registered by
    installed third-party packages through the quakemigrate.plugins entry-point group.

    Returns
    -------
    plugins:
        Mapping from plugin name to plugin class or factory function.

    Raises
    ------
    ConfigError
        If a built-in plugin cannot be loaded, an entry point cannot be loaded, or a
        third-party entry point uses the same name as a built-in plugin.

    """

    plugins = {
        name: load_builtin_plugin(name, import_path)
        for name, import_path in BUILTIN_PLUGINS.items()
    }

    for name, target in get_entry_point_plugins().items():
        if name in plugins:
            raise ConfigError(
                f"Plugin name {name} is defined by both a built-in plugin and an"
                "installed entry point."
            )

        plugins[name] = target

    return plugins


def construct_plugin(
    plugin_config: Mapping[str, Any],
    **context: Any,
) -> Any:
    """
    Construct a plugin object from a plugin specification.

    The specification must contain name, which may refer to either a built-in plugin or
    a third-party plugin registered through the quakemigrate.plugins entry-point group.

    Any remaining keys in plugin_config are treated as configuration arguments and are
    injected into the target constructor or factory function by signature, together with
    any additional runtime dependencies supplied through context.

    Parameters
    ----------
    plugin_config:
        Plugin specification dictionary. Must contain the key "name".
    **context:
        Additional dependencies made available for signature-based injection.

    Returns
    -------
    plugin:
        Constructed plugin object.

    Raises
    ------
    ConfigError
        If the specification is invalid, the plugin name is unknown, the plugin target
        cannot be loaded, required dependencies are missing, or construction fails.

    """

    plugin_config = dict(plugin_config)

    plugin_name = plugin_config.pop("name", None)

    if plugin_name is None:
        raise ConfigError("Plugin specification must contain 'name'.")

    plugins = get_available_plugins()

    try:
        target = plugins[plugin_name]
    except KeyError as e:
        allowed = ", ".join(sorted(plugins))
        raise ConfigError(
            f"Unknown plugin {plugin_name}. Expected one of: [{allowed}]."
        ) from e

    plugin = call_by_signature(target, {**plugin_config, **context})

    if not isinstance(plugin, Plugin):
        raise ConfigError(
            f"Constructed plugin {plugin} does not conform to the plugin protocol."
        )

    return plugin
