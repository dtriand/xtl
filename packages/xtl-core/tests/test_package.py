import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import tomlkit

import xtl_core
from xtl_core import version as version_info
from xtl_core.version_info import VersionInfo

ROOT = Path(__file__).parents[1]


class TestVersion:
    def test_version_info(self):
        assert isinstance(version_info, VersionInfo)
        assert xtl_core.__version__ == version_info.pep440

    def test_fits_in_hex(self):
        # The serial is a single hex digit
        assert 0 <= version_info.serial < 16
        assert all(0 <= n < 256 for n in version_info.tuple_safe)
        assert VersionInfo.from_hex(version_info.hex, version_info.release_date or None) == version_info

    def test_matches_metadata(self):
        # Stale if the version changed since the last `uv sync`
        assert xtl_core.__version__ == version('xtl-core')

    def test_hatch_pattern(self):
        # hatch reads the version with this regex and writes it into the metadata as is, so
        #  it must find exactly one string, already in its PEP 440 form
        pyproject = tomlkit.parse((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        config = pyproject['tool']['hatch']['version']
        source = (ROOT / config['path']).read_text(encoding='utf-8')
        assert re.findall(config['pattern'], source) == [version_info.pep440]


def test_import_does_not_import_xtl():
    code = 'import sys, xtl_core; sys.exit(int("xtl" in sys.modules))'
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


def test_library_logging():
    import logging

    handlers = logging.getLogger('xtl_core').handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.NullHandler)
