"""
Pydantic models with extras: options with validation and casting, serialization to
dicts, JSON and TOML, and versioned data layouts.
"""

from xtl_core.options.base import Option, Options, VersionedOptions

__all__ = ['Option', 'Options', 'VersionedOptions']
