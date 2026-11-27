"""
Pydantic models with extras: options with validation and casting, serialization to
dicts, JSON and TOML, and versioned data layouts.
"""

from xtl_core.options.base import Option, Options
from xtl_core.options.exceptions import OptionsValidationError
from xtl_core.options.versioned import VersionedOptions

__all__ = ['Option', 'Options', 'OptionsValidationError', 'VersionedOptions']
