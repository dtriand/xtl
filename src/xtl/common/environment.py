"""
Central location for environment variables and related utilities.
"""
import os

#########################
# Environment variables #
#########################
XTL_COMPUTE_SITE: str | None = os.getenv('XTL_COMPUTE_SITE', None)
"""Environment variable 'XTL_COMPUTE_SITE'"""
# Used to trigger specialized compute site configurations

XTL_FONTS_DIR: str | None = os.getenv('XTL_FONTS_DIR', None)
"""Environment variable 'XTL_FONTS_DIR'"""
# Used to load fonts from custom directories
