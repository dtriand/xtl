import os
from pathlib import Path
import subprocess
import sysconfig
from importlib.metadata import distribution

import pytest

from xtl_core.compatibility import OS_WINDOWS


CONSOLE_SCRIPTS = sorted(ep.name for ep in distribution('xtl').entry_points if ep.group == 'console_scripts')
SCRIPTS_DIR = Path(sysconfig.get_path('scripts'))


def test_console_scripts_are_installed():
    assert CONSOLE_SCRIPTS, 'No console scripts found for the installed xtl distribution'


@pytest.mark.parametrize('name', CONSOLE_SCRIPTS)
def test_console_script_help(name):
    script = SCRIPTS_DIR / (f'{name}.exe' if OS_WINDOWS else name)
    assert script.exists(), f'Console script not found: {script}'

    # Help texts contain non-ASCII characters (e.g. 2θ), which can't be encoded on
    # Windows when stdout is redirected and the console encoding is not UTF-8
    env = os.environ | {'PYTHONUTF8': '1'}
    result = subprocess.run([script, '--help'], capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
    assert result.returncode == 0, f'{name} --help failed:\n{result.stderr}'
    assert 'Usage' in result.stdout
