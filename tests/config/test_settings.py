import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from xtl.config.settings import XTLSettings


NAMES = ['env', 'local', 'user', 'global']


def _records(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == level and r.name == 'xtl.config.settings']


def _path(locs, name: str) -> Path:
    return {'env': locs.env, 'local': locs.local, 'user': locs.user,
            'global': locs.global_}[name]


def test_session_uses_temp_config():
    # tests/conftest.py points XTL_CONFIG to a temporary file for the whole session
    from xtl import settings

    config = Path(os.environ['XTL_CONFIG'])
    assert config.name == 'xtl.toml'
    assert config.parent.name.startswith('xtl-tests-')
    assert settings.source is not None
    assert settings.source.name == 'env'
    assert settings.source.path == config


class TestLocations:

    def test_without_env(self, locs):
        locations = XTLSettings.locations()
        assert [loc.name for loc in locations] == ['local', 'user', 'global']
        assert [loc.path for loc in locations] == [locs.local, locs.user, locs.global_]
        assert not any(loc.exists for loc in locations)

    def test_with_env(self, locs):
        locs.set_env()
        locations = XTLSettings.locations()
        assert [loc.name for loc in locations] == NAMES
        assert locations[0].path == locs.env

    def test_creatable(self, locs):
        locs.set_env()
        assert {loc.name: loc.creatable for loc in XTLSettings.locations()} == \
               {'env': True, 'local': False, 'user': True, 'global': True}

    def test_env_directory_is_ignored(self, locs, caplog):
        env_dir = locs.root / 'env'
        env_dir.mkdir()
        locs.set_env(env_dir)
        with caplog.at_level(logging.WARNING):
            locations = XTLSettings.locations()
        assert [loc.name for loc in locations] == ['local', 'user', 'global']
        assert any('points to a directory' in m
                   for m in _records(caplog, logging.WARNING))

    def test_env_is_read_on_every_call(self, locs):
        assert 'env' not in [loc.name for loc in XTLSettings.locations()]
        locs.set_env()
        assert XTLSettings.locations()[0].name == 'env'

    def test_user_path(self, monkeypatch):
        # appauthor=False, otherwise Windows gives ...\xtl\xtl
        calls = []
        monkeypatch.setattr('platformdirs.user_config_dir',
                            lambda *args, **kwargs: calls.append((args, kwargs)) or '.')
        XTLSettings.locations()
        assert calls == [(('xtl',), {'appauthor': False})]

    def test_exists_is_a_file_check(self, locs):
        locs.local.mkdir()
        assert not XTLSettings.locations()[0].exists


class TestActive:

    def test_none(self, locs):
        assert XTLSettings.active() is None

    def test_first_existing(self, locs):
        locs.write(locs.user)
        locs.write(locs.global_)
        assert XTLSettings.active().name == 'user'

    def test_given_locations(self, locs):
        locs.write(locs.user)
        locations = XTLSettings.locations()
        locs.write(locs.local)
        # Uses the given locations, but checks existence now
        assert XTLSettings.active(locations).name == 'local'
        assert XTLSettings.active(locations[2:]) is None


class TestInitialize:

    @pytest.mark.parametrize('name', NAMES)
    def test_precedence(self, locs, name):
        # A file in a location beats the files in all later locations
        locs.set_env()
        for n in NAMES[NAMES.index(name):]:
            locs.write(_path(locs, n), f'[units]\ntemperature = "K"\n'
                       if n == name else '[units]\ntemperature = "F"\n')
        before = locs.files()

        s = XTLSettings.initialize()
        assert s.source.name == name
        assert s.source.path == _path(locs, name)
        assert s.units.temperature == 'K'
        assert locs.files() == before

    def test_create_in_env(self, locs, caplog):
        locs.set_env()
        with caplog.at_level(logging.WARNING):
            s = XTLSettings.initialize()
        assert s.source.name == 'env'
        assert locs.files() == {Path('env/config.cfg')}  # file extension kept
        assert not _records(caplog, logging.WARNING)

    def test_create_in_user(self, locs):
        s = XTLSettings.initialize()
        assert s.source.name == 'user'
        assert locs.files() == {Path('user/xtl.toml')}  # never local

    def test_create_falls_through(self, locs, caplog):
        locs.block(locs.user)
        with caplog.at_level(logging.WARNING):
            s = XTLSettings.initialize()
        assert s.source.name == 'global'
        assert locs.global_.is_file()
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert 'Failed to create user config' in warnings[0]

    def test_create_all_fail(self, locs, caplog):
        locs.set_env()
        for path in (locs.env, locs.user, locs.global_):
            locs.block(path)
        before = locs.files()
        with caplog.at_level(logging.WARNING):
            s = XTLSettings.initialize()
        assert s.source is None
        assert s.to_dict() == XTLSettings().to_dict()
        assert locs.files() == before
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 4  # one per location, plus one for using the defaults
        assert 'in-memory' in warnings[-1]

    def test_no_create(self, locs):
        s = XTLSettings.initialize(create=False)
        assert s.source is None
        assert locs.files() == set()

    def test_created_file_reads_back(self, locs, caplog):
        locs.set_env()
        XTLSettings.initialize()
        with caplog.at_level(logging.WARNING):
            s = XTLSettings.initialize()
        assert s.source.path == locs.env
        assert s.to_dict() == XTLSettings().to_dict()
        assert not _records(caplog, logging.WARNING)  # no version or unknown keys

    def test_env_round_trip(self, locs):
        # A non-.toml XTL_CONFIG is created and read under its own name
        locs.set_env()
        assert XTLSettings.initialize().source.path == locs.env
        locs.write(locs.env, '[units]\ntemperature = "K"\n')
        assert XTLSettings.initialize().units.temperature == 'K'
        assert locs.files() == {Path('env/config.cfg')}

    @pytest.mark.parametrize('text', [
        pytest.param('[units\n', id='invalid toml'),
        pytest.param('[units]\ntemperature = "X"\n', id='invalid value'),
    ])
    def test_broken_file(self, locs, caplog, text):
        # D-021: defaults, the file isn't overwritten, later locations are ignored
        locs.write(locs.local, text)
        locs.write(locs.user, '[units]\ntemperature = "K"\n')
        before = locs.files()
        with caplog.at_level(logging.WARNING):
            s = XTLSettings.initialize()
        assert s.source is None
        assert s.units.temperature == 'C'
        assert locs.local.read_text(encoding='utf-8') == text
        assert locs.files() == before
        errors = _records(caplog, logging.ERROR)
        assert len(errors) == 1
        assert str(locs.local) in errors[0]

    def test_warnings_are_logged(self, locs, caplog):
        locs.write(locs.local, '[units]\ncolour = "blue"\n')
        with caplog.at_level(logging.WARNING):
            s = XTLSettings.initialize()
        assert s.source.name == 'local'
        assert _records(caplog, logging.WARNING) == \
               ['Found and ignored unknown keys in the config file: units.colour']


@pytest.mark.xfail(strict=True, reason='Q-015: nested defaults are shared, because '
                   '`Settings` is hashable and pydantic does not copy hashable '
                   'defaults')
def test_defaults_are_not_shared():
    a, b = XTLSettings(), XTLSettings()
    assert a.units is not b.units


class TestCheck:

    def test_defaults(self):
        assert XTLSettings().check() == []

    @pytest.mark.parametrize('version, message', [
        ('0.0.1', 'older version'),
        ('99.0.0', 'newer version'),
        ('not-a-version', 'Invalid settings version'),
        ('0.1', 'Invalid settings version'),
    ])
    def test_version(self, version, message):
        problems = XTLSettings(version=version).check()
        assert len(problems) == 1
        assert message in problems[0]

    def test_unknown_keys(self):
        s = XTLSettings.from_toml('colour = "blue"\n'
                                  '[units]\ncolour = "blue"\n'
                                  '[jobs.resources]\nfoo = 1\nbar = 2\n')
        problems = s.check()
        assert len(problems) == 1
        assert problems[0].endswith(': colour, units.colour, jobs.resources.foo, '
                                    'jobs.resources.bar')

    def test_does_not_mutate(self):
        s = XTLSettings.from_toml('[units]\ncolour = "blue"\n')
        extra = dict(s.units.model_extra)
        assert s.check() == s.check()
        assert s.units.model_extra == extra


class TestCheckFile:

    def test_valid(self, locs):
        path = locs.write(locs.local, XTLSettings().to_toml())
        assert XTLSettings.check_file(path) == ([], [])

    def test_warnings(self, locs):
        path = locs.write(locs.local, f'version = "0.0.1"\n[units]\ncolour = "blue"\n')
        errors, warnings = XTLSettings.check_file(path)
        assert errors == []
        assert len(warnings) == 2

    def test_invalid_values(self, locs):
        path = locs.write(locs.local, '[units]\ntemperature = "X"\n'
                                      '[cli]\nmax_fps = 1000\n')
        errors, warnings = XTLSettings.check_file(path)
        assert warnings == []
        assert len(errors) == 2
        # Located without the model name
        assert errors[0].startswith('units.temperature: ')
        assert errors[1].startswith('cli.max_fps: ')

    def test_invalid_toml(self, locs):
        path = locs.write(locs.local, '[units\n')
        errors, warnings = XTLSettings.check_file(path)
        assert len(errors) == 1
        assert warnings == []

    def test_missing(self, locs):
        errors, warnings = XTLSettings.check_file(locs.local)
        assert len(errors) == 1
        assert warnings == []


class TestLazyImport:

    @staticmethod
    def _run(code: str) -> str:
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    @pytest.mark.parametrize('module', ['xtl', 'xtl.cli.config'])
    def test_import_does_not_load_settings(self, module):
        out = self._run(f'import sys, xtl, {module}\n'
                        f'print("xtl.config.settings" in sys.modules, '
                        f'"settings" in vars(xtl))')
        assert out == 'False False'

    def test_import_xtl_is_cheap(self):
        out = self._run('import sys, xtl\n'
                        'print(any(m in sys.modules for m in ("xtl.automate", '
                        '"xtl.jobs")))')
        assert out == 'False'

    def test_settings_is_cached(self):
        out = self._run('from xtl import settings as a\n'
                        'from xtl import settings as b\n'
                        'import xtl\n'
                        'print(a is b is xtl.settings, type(a).__name__)')
        assert out == 'True XTLSettings'
