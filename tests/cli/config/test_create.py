from pathlib import Path

import pytest
import toml

from tests.cli.config.conftest import INVALID, VALID, defaults
from xtl.config.settings import XTLSettings


class TestCreate:

    def test_default_is_local(self, cli, locs):
        result = cli('create', writes=True)
        assert result.exit_code == 0, result.output
        assert locs.files() == {Path('cwd/xtl.toml')}
        assert 'from the default settings (no config file found)' in result.stdout
        assert toml.loads(locs.local.read_text(encoding='utf-8')) == defaults()
        assert XTLSettings.check_file(locs.local) == ([], [])
        # The new file is the active one: no notes
        assert 'Note:' not in result.stdout
        assert 'not in the lookup path' not in result.stdout

    @pytest.mark.parametrize('args, name', [
        (['--local'], 'local'),
        (['--user'], 'user'),
        (['--global'], 'global'),
        (['--env'], 'env'),
        (['-l', 'user'], 'user'),
        (['--location', 'GLOBAL'], 'global'),
        (['--location', 'env'], 'env'),
        (['-l', 'user', '--user'], 'user'),  # same location twice
    ])
    def test_location(self, cli, locs, args, name):
        locs.set_env()
        path = {'env': locs.env, 'local': locs.local, 'user': locs.user,
                'global': locs.global_}[name]
        result = cli('create', *args, writes=True)
        assert result.exit_code == 0, result.output
        assert locs.files() == {path.relative_to(locs.root)}

    def test_path(self, cli, locs):
        path = locs.root / 'custom' / 'my.cfg'
        result = cli('create', '--path', str(path), writes=True)
        assert result.exit_code == 0, result.output
        assert locs.files() == {Path('custom/my.cfg')}  # file extension kept
        assert 'not in the lookup path' in result.stdout

    @pytest.mark.parametrize('args', [
        ['--user', '--global'],
        ['--local', '--path', 'x.toml'],
        ['-l', 'user', '--global'],
    ])
    def test_conflicting_locations(self, cli, args):
        result = cli('create', *args)
        assert result.exit_code == 2
        assert 'mutually exclusive' in result.output

    def test_env_not_set(self, cli):
        result = cli('create', '--env')
        assert result.exit_code == 2
        assert 'XTL_CONFIG is not set' in result.output

    def test_env_directory(self, cli, locs):
        env_dir = locs.root / 'env'
        env_dir.mkdir()
        locs.set_env(env_dir)
        result = cli('create', '--env')
        assert result.exit_code == 2
        assert 'points to a directory' in result.output

    def test_exists(self, cli, locs):
        locs.write(locs.local, VALID)
        result = cli('create', '--clean')
        assert result.exit_code == 1
        assert '--force' in result.stdout
        assert locs.local.read_text(encoding='utf-8') == VALID

    def test_force(self, cli, locs):
        locs.write(locs.local, VALID)
        result = cli('create', '--clean', '--force', writes=True)
        assert result.exit_code == 0, result.output
        assert toml.loads(locs.local.read_text(encoding='utf-8')) == defaults()

    def test_copies_active(self, cli, locs):
        locs.write(locs.user, VALID + 'colour = "blue"\n')
        result = cli('create', writes=True)
        assert result.exit_code == 0, result.output
        assert 'from the user config' in result.stdout
        data = toml.loads(locs.local.read_text(encoding='utf-8'))
        assert data['units'] == {'temperature': 'K'}  # unknown keys dropped
        assert data == defaults() | {'units': {'temperature': 'K'}}

    def test_clean(self, cli, locs):
        locs.write(locs.user, VALID)
        result = cli('create', '--clean', writes=True)
        assert result.exit_code == 0, result.output
        assert 'from the default settings' in result.stdout
        assert toml.loads(locs.local.read_text(encoding='utf-8')) == defaults()

    def test_invalid_active(self, cli, locs):
        locs.write(locs.user, INVALID)
        result = cli('create')
        assert result.exit_code == 1
        assert 'units.temperature' in result.stdout
        assert '--clean' in result.stdout

    def test_invalid_active_clean(self, cli, locs):
        locs.write(locs.user, INVALID)
        result = cli('create', '--clean', writes=True)
        assert result.exit_code == 0, result.output
        assert locs.local.is_file()

    def test_write_failure(self, cli, locs):
        locs.write(locs.root / 'file')
        result = cli('create', '--path', str(locs.root / 'file' / 'xtl.toml'))
        assert result.exit_code == 1
        assert 'failed to write' in result.stdout

    def test_shadowed(self, cli, locs):
        locs.write(locs.local, VALID)
        result = cli('create', '--user', writes=True)
        assert result.exit_code == 0, result.output
        assert locs.user.is_file()
        assert 'the local config' in result.stdout
        assert 'takes precedence over this file' in result.stdout
