from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
import os
import random
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

from xtl_core.system.compatibility import OS_POSIX, OS_WINDOWS

CACHE_DIR = Path(__file__).parent / 'cache'

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()
else:
    shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir()

# Config file used by `xtl.settings` during the test session (see `pytest_configure`)
_SETTINGS_DIR: Path | None = None


def pytest_configure(config):
    """
    Point `XTL_CONFIG` to an empty (all defaults) temporary file, so that the tests
    never read or create a config file in the working directory, the user config
    directory or the package folder. The file must exist, otherwise any existing
    config would be loaded instead. Set before collection, since importing some
    modules loads `xtl.settings`. Subprocesses inherit it.
    """
    global _SETTINGS_DIR
    _SETTINGS_DIR = Path(tempfile.mkdtemp(prefix='xtl-tests-'))
    config = _SETTINGS_DIR / 'xtl.toml'
    config.touch()
    os.environ['XTL_CONFIG'] = str(config)


def pytest_unconfigure(config):
    if _SETTINGS_DIR is not None:
        shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


@pytest.fixture(scope='session', params=[f'{random.randint(0, 9999):04d}' for _ in range(0, 3)])
def seed(request):
    """
    Creates 3 four-digit integer seeds and passes them to functions decorated with this fixture
    """
    return request.param


@pytest.fixture
def temp_files(request, tmp_path_factory):
    # Get marker
    marker = request.node.get_closest_marker('make_temp_files')
    params = getattr(request, 'param', None)

    if marker:
        files = marker.args
    else:
        if params is None:
            raise ValueError("No 'make_temp_files' marker found and no 'param' provided on request")
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
skipif_not_wsl = pytest.mark.skipif(shutil.which('wsl') is None, reason='WSL not installed')

supported_distros = ['Ubuntu-18.04', 'Ubuntu-22.04']


@cache
def wsl_distro_exists(distro):
    """
    Check if a WSL distro exists via the `wsl -d <distro> -e true` command
    """
    return subprocess.run(f'wsl -d {distro} -e true').returncode == 0


@contextmanager
def random_seed(n: int) -> None:
    """
    Set the random seed for a test
    """
    random.seed(n)
    yield
    random.seed()


@dataclass
class Locations:
    """
    Isolated config locations for a test: ``env`` is the path that ``XTL_CONFIG`` is set
    to by :meth:`set_env` (unset by default).
    """

    root: Path
    env: Path
    local: Path
    user: Path
    global_: Path
    monkeypatch: pytest.MonkeyPatch

    def set_env(self, path: Path = None) -> Path:
        """Set ``XTL_CONFIG`` to ``path`` (default: :attr:`env`)."""
        path = self.env if path is None else path
        self.monkeypatch.setenv('XTL_CONFIG', str(path))
        return path

    def write(self, path: Path, text: str = '') -> Path:
        """Write a config file, creating its parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def block(self, path: Path) -> None:
        """Make ``path`` impossible to create, by turning its parent into a file."""
        self.write(path.parent)

    def files(self) -> set[Path]:
        """All files under :attr:`root`, relative to it."""
        return {p.relative_to(self.root) for p in self.root.rglob('*') if p.is_file()}


@pytest.fixture
def locs(tmp_path, monkeypatch) -> Locations:
    """
    Config locations inside ``tmp_path``: the working directory, the user config
    directory and the global config are patched, and ``XTL_CONFIG`` is unset.
    """
    root = tmp_path
    cwd = root / 'cwd'
    user = root / 'user'
    cwd.mkdir()

    from xtl.config.settings import XTLSettings

    monkeypatch.delenv('XTL_CONFIG', raising=False)
    monkeypatch.chdir(cwd)
    # Imported inside XTLSettings.locations(), so patch it at the source
    monkeypatch.setattr('platformdirs.user_config_dir', lambda *args, **kwargs: str(user))
    monkeypatch.setattr(XTLSettings, 'global_config', root / 'global' / 'xtl.toml')

    return Locations(
        root=root,
        env=root / 'env' / 'config.cfg',
        local=cwd / 'xtl.toml',
        user=user / 'xtl.toml',
        global_=root / 'global' / 'xtl.toml',
        monkeypatch=monkeypatch,
    )
