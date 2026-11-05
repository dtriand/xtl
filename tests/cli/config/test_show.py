import json

import toml

from tests.cli.config.conftest import INVALID, VALID, defaults
from xtl.config.settings import XTLSettings


class TestShow:
    @staticmethod
    def _show(cli, *args: str, code: int = 0):
        result = cli('show', *args)
        assert result.exit_code == code, result.output
        return result

    def test_defaults(self, cli):
        out = self._show(cli).stdout
        assert out.startswith('# Source: in-memory defaults\n')
        assert toml.loads(out) == defaults()
        assert '# C for Celsius' in out  # field descriptions

    def test_source(self, cli, locs):
        locs.write(locs.user, VALID)
        out = self._show(cli).stdout
        assert out.startswith(f'# Source: user ({locs.user})\n')
        assert toml.loads(out)['units'] == {'temperature': 'K'}

    def test_section(self, cli):
        assert toml.loads(self._show(cli, 'units').stdout) == {'units': {'temperature': 'C'}}
        data = toml.loads(self._show(cli, 'dependencies.atsas').stdout)
        assert list(data) == ['dependencies']
        assert list(data['dependencies']) == ['atsas']

    def test_key(self, cli):
        # Without the parent tables
        assert toml.loads(self._show(cli, 'units.temperature').stdout) == {'temperature': 'C'}

    def test_unknown_key(self, cli):
        result = self._show(cli, 'units.temp', code=2)
        assert 'Unknown key: units.temp' in result.output
        assert 'Did you mean' in result.output
        assert 'Did you mean: units.temperature?' in result.output

    def test_unknown_table(self, cli):
        # Only keys at the same depth are suggested, not the keys of the table
        result = self._show(cli, 'dependecies', code=2)
        assert 'Did you mean: dependencies?' in result.output

    def test_unknown_key_in_file(self, cli, locs):
        # Unknown keys are not shown, so they can't be selected either
        locs.write(locs.local, '[units]\ncolour = "blue"\n')
        self._show(cli, 'units.colour', code=2)
        assert 'colour' not in self._show(cli).stdout

    def test_key_not_set(self, cli):
        out = self._show(cli, 'dependencies.atsas.path').stdout
        assert out.splitlines()[1] == '# dependencies.atsas.path is not set'

    def test_diff(self, cli, locs):
        locs.write(locs.local, VALID)
        out = self._show(cli, '--diff').stdout
        assert toml.loads(out) == {'units': {'temperature': 'K'}}
        assert '# default: "C"' in out
        assert '# default:' not in self._show(cli, '--diff', '--no-comments').stdout

    def test_diff_key(self, cli, locs):
        locs.write(locs.local, VALID + '[cli]\nmax_fps = 20\n')
        out = self._show(cli, '--diff', 'cli').stdout
        assert toml.loads(out) == {'cli': {'max_fps': 20}}
        assert '# default: 10' in out  # not a string

    def test_diff_none(self, cli, locs):
        out = self._show(cli, '--diff').stdout
        assert '# Same as the defaults' in out
        assert toml.loads(out) == {}

    def test_diff_unset_default(self, cli, locs):
        # Unset by default, set in the file
        locs.write(locs.local, '[dependencies.atsas]\npath = "/opt/atsas"\n')
        out = self._show(cli, '--diff').stdout
        assert toml.loads(out) == {'dependencies': {'atsas': {'path': '/opt/atsas'}}}
        assert '# default: not set' in out

    def test_file(self, cli, locs):
        locs.write(locs.local, VALID)
        path = locs.write(locs.root / 'other.cfg', '[units]\ntemperature = "F"\n')
        out = self._show(cli, '--file', str(path)).stdout
        assert out.startswith(f'# Source: {path}\n')
        assert toml.loads(out)['units'] == {'temperature': 'F'}

    def test_file_invalid(self, cli, locs):
        path = locs.write(locs.root / 'other.toml', INVALID)
        result = self._show(cli, '--file', str(path), code=1)
        assert 'invalid config file' in result.stderr
        assert result.stdout == ''

    def test_file_missing(self, cli, locs):
        self._show(cli, '--file', str(locs.root / 'missing.toml'), code=2)

    def test_invalid_active(self, cli, locs):
        locs.write(locs.local, INVALID)
        result = self._show(cli)
        assert 'Warning:' in result.stderr
        assert result.stdout.startswith('# Source: in-memory defaults (local config is invalid)\n')
        assert toml.loads(result.stdout) == defaults()

    def test_json(self, cli, locs):
        locs.write(locs.local, VALID)
        data = json.loads(self._show(cli, '--json').stdout)
        assert data['units'] == {'temperature': 'K'}
        assert data['version'] == XTLSettings().version
        assert json.loads(self._show(cli, '--json', 'units').stdout) == {'temperature': 'K'}
        assert json.loads(self._show(cli, '--json', 'units.temperature').stdout) == 'K'
        assert json.loads(self._show(cli, '--json', '--diff').stdout) == {'units': {'temperature': 'K'}}

    def test_no_comments(self, cli):
        lines = self._show(cli, '--no-comments').stdout.splitlines()
        assert lines[0].startswith('# Source:')
        assert not any('#' in line for line in lines[1:])
