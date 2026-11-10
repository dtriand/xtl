import subprocess
import sys
from importlib.metadata import version

import xtl_core


def test_version_matches_metadata():
    assert xtl_core.__version__ == version('xtl-core')


def test_import_does_not_import_xtl():
    code = 'import sys, xtl_core; sys.exit(int("xtl" in sys.modules))'
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


def test_library_logging():
    import logging

    handlers = logging.getLogger('xtl_core').handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.NullHandler)
