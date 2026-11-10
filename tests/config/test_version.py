import re
from importlib.metadata import version as dist_version
from pathlib import Path

import pytest

import xtl
from xtl.config.version import version
from xtl_core.version import VersionInfo

ROOT = Path(__file__).parents[2]


class TestPackageVersion:
    def test_exported(self):
        assert xtl.version is version
        assert isinstance(version, VersionInfo)

    def test_fits_in_hex(self):
        # The serial is a single hex digit
        assert 0 <= version.serial < 16
        assert all(0 <= n < 256 for n in version.tuple_safe)
        assert VersionInfo.from_hex(version.hex, version.release_date) == version

    def test_round_trip(self):
        assert VersionInfo.from_str(version.string, version.release_date) == version

    def test_distribution_metadata(self):
        # hatch normalizes the string it reads to PEP 440 (stale if the version changed
        #  since the last `uv sync`)
        assert dist_version('xtl') == version.pep440

    def test_hatch_pattern(self):
        # hatch reads the version with this regex; it must find exactly the string that
        #  `version` is built from
        tomllib = pytest.importorskip('tomllib')  # Python 3.11+
        pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        config = pyproject['tool']['hatch']['version']
        source = (ROOT / config['path']).read_text(encoding='utf-8')
        matches = re.findall(config['pattern'], source)
        assert matches == [version.string]
