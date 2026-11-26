import getpass
import os
import platform
import sys
import tempfile
import types

import pytest

from xtl_core.system import os as xtl_os
from xtl_core.system.os import CPU_CORES, get_os_name_and_version, get_temp_location, get_username, is_temp


def test_cpu_cores():
    assert CPU_CORES == os.cpu_count()


class TestGetOsNameAndVersion:
    def test_not_linux(self, monkeypatch):
        monkeypatch.setattr(platform, 'system', lambda: 'Windows')
        monkeypatch.setattr(platform, 'version', lambda: '10.0.26300')
        assert get_os_name_and_version() == 'Windows 10.0.26300'

    def test_linux(self, monkeypatch):
        distro = types.SimpleNamespace(name=lambda: 'Ubuntu', version=lambda: '24.04')
        monkeypatch.setattr(platform, 'system', lambda: 'Linux')
        monkeypatch.setitem(sys.modules, 'distro', distro)
        assert get_os_name_and_version() == 'Ubuntu 24.04'

    def test_linux_without_distro(self, monkeypatch):
        # A `None` entry in `sys.modules` makes the import fail
        monkeypatch.setattr(platform, 'system', lambda: 'Linux')
        monkeypatch.setattr(platform, 'version', lambda: '#1 SMP')
        monkeypatch.setitem(sys.modules, 'distro', None)
        assert get_os_name_and_version() == 'Linux #1 SMP'

    def test_this_system(self):
        assert get_os_name_and_version()


class TestGetUsername:
    def test_login(self, monkeypatch):
        monkeypatch.setattr(os, 'getlogin', lambda: 'someone')
        assert get_username() == 'someone'

    def test_no_terminal(self, monkeypatch):
        # `os.getlogin()` fails without a controlling terminal (CI, services)
        def getlogin():
            raise OSError('no terminal')

        monkeypatch.setattr(os, 'getlogin', getlogin)
        monkeypatch.setattr(getpass, 'getuser', lambda: 'runner')
        assert get_username() == 'runner'


class TestGetTempLocation:
    def test_tempfile(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
        assert get_temp_location() == tmp_path


class TestIsTemp:
    @pytest.fixture
    def temp(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
        return tmp_path

    @pytest.mark.parametrize('name', ['file.txt', 'a/b/file.txt'], ids=['file', 'nested'])
    def test_inside(self, temp, name):
        assert is_temp(temp / name)
        assert is_temp(str(temp / name))

    def test_temp_itself(self, temp):
        assert not is_temp(temp)

    def test_outside(self, temp):
        assert not is_temp(temp.parent / 'elsewhere.txt')

    def test_relative(self, temp, monkeypatch):
        monkeypatch.chdir(temp)
        assert is_temp('file.txt')

    def test_temp_not_resolved(self, monkeypatch, tmp_path):
        # As `/var/folders/...` (macOS, really `/private/var/...`) or `RUNNER~1` (Windows short names)
        (tmp_path / 'sub').mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'sub' / '..'))
        assert is_temp(tmp_path / 'file.txt')


def test_exports():
    for name in xtl_os.__all__:
        assert hasattr(xtl_os, name)
