"""
``xtl.core`` is ``xtl_core`` (the alias itself is tested in xtl-core's own tests).
"""

import subprocess
import sys

import xtl_core


def test_core_is_xtl_core():
    import xtl.core

    assert xtl.core is xtl_core
    assert sys.modules['xtl.core'] is xtl_core


def test_submodules_are_the_same():
    import xtl.core._version
    import xtl_core._version

    assert xtl.core._version is xtl_core._version
    assert xtl.core._version.__name__ == 'xtl_core._version'


def test_from_import():
    from xtl.core import __version__

    assert __version__ == xtl_core.__version__


def test_import_is_lazy():
    # In a new interpreter, so that other tests' imports don't count
    code = (
        'import sys, xtl.core; '
        "print(sorted(m for m in sys.modules if m == 'xtl_core' or m.startswith(('xtl_core.', 'xtl.core'))))"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "['xtl.core', 'xtl_core', 'xtl_core._alias', 'xtl_core._version']"
