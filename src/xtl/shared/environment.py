"""
xtl's environment variables (see :class:`~xtl_core.environment.EnvVar`).
"""

from pathlib import Path

from xtl_core.environment import EnvVar, expand_path

#########################
# Environment variables #
#########################
XTL_CONFIG: EnvVar[Path] = EnvVar('XTL_CONFIG', parser=expand_path)
"""
Path to the config file, which takes precedence over all other config file locations
(see :meth:`XTLSettings.locations() <xtl.config.settings.XTLSettings.locations>`).
"""

XTL_COMPUTE_SITE: EnvVar[str] = EnvVar('XTL_COMPUTE_SITE')
"""
Enables compute sites (see :class:`xtl.jobs.sites.ComputeSite`) for specific facilities.
Read when :mod:`xtl.jobs.sites` is imported. Currently supported values are:

- ``BIOTIX_HPC`` enables ``biotix``
"""

XTL_FONTS_DIR: EnvVar[Path] = EnvVar('XTL_FONTS_DIR', parser=expand_path)
"""
Directory with additional fonts for plots. Reserved for the plotting backends, which
are still in development; it has no effect yet.
"""
