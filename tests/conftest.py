from contextlib import contextmanager
from functools import cache
import random
from pathlib import Path
import shutil
import subprocess

import pytest

from xtl.common.compatibility import OS_POSIX, OS_WINDOWS

CACHE_DIR = Path(__file__).parent / 'cache'

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()
else:
    shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir()


@pytest.fixture(scope='session', params=[f'{random.randint(0, 9999):04d}' for _ in range(0, 3)])
def seed(request):
    """
    Creates 3 four-digit integer seeds and passes them to functions decorated with this fixture
    """
    return request.param


@pytest.fixture
def temp_files(request, tmp_path_factory):
    # Get marker
    marker = request.node.get_closest_marker("make_temp_files")
    params = getattr(request, 'param', None)

    if marker:
        files = marker.args
    else:
        if params is None:
            raise ValueError('No \'make_temp_files\' marker found and no \'param\' provided on request')
        elif isinstance(params, list | tuple):
            files = request.param
        else:
            files = (request.param,)

    # Create a temporary directory to store the files
    temp_dir = tmp_path_factory.mktemp('cached')

    # Create all files and directories
    temp_files: list[Path] = []
    for f in files:
        if f is None:
            temp_files.append(None)
            continue
        f = temp_dir / f
        if (not f.suffix) and (f.name != ''):  # check if f is a directory because f.is_dir() fails when f doesn't exist
            f.mkdir(parents=True, exist_ok=True)
            print(f'Created empty temporary directory: {f}')
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
            print(f'Created empty temporary file: {f}')
        temp_files.append(f)

    # Return the temporary files as Path objects
    if len(temp_files) == 1:
        return temp_files[0]
    return temp_files


skipif_not_windows = pytest.mark.skipif(not OS_WINDOWS, reason='Test only for Windows')
skipif_not_linux = pytest.mark.skipif(not OS_POSIX, reason='Test only for Linux')
skipif_not_wsl = pytest.mark.skipif(shutil.which("wsl") is None, reason='WSL not installed')

supported_distros = ['Ubuntu-18.04', 'Ubuntu-22.04']

@cache
def wsl_distro_exists(distro):
    """
    Check if a WSL distro exists via the `wsl -d <distro> -e true` command
    """
    return subprocess.run(f'wsl -d {distro} -e true').returncode == 0


@contextmanager
def seed(n: int) -> None:
    """
    Set the random seed for a test
    """
    random.seed(n)
    yield
    random.seed()