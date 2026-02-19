"""
Compatibility shim.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import warnings

from quakemigrate.plugins.pickers.gaussian import GaussianPicker


warnings.warn(
    "quakemigrate.signal.pickers.gaussian is deprecated. "
    "Use quakemigrate.plugins.pickers.gaussian instead.",
    DeprecationWarning,
    stacklevel=2,
)
