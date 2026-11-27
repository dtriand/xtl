import pytest

from tests.cli.config.conftest import INVALID

from xtl.config.settings import XTLSettings
from xtl.config.version import version

OLD = '# my settings\nversion = "0.0.1" # mine\n\n[units]\ntemperature = "K" # kelvin\n'


def upgraded(text: str) -> str:
    return text.replace('"0.0.1"', f'"{version.pep440}"')


class TestUpgrade:
    def test_no_active(self, cli):
        result = cli('upgrade')
        assert result.exit_code == 0
        assert 'nothing to upgrade' in result.stdout

    def test_active(self, cli, locs):
        path = locs.write(locs.user, OLD)
        result = cli('upgrade', writes=True)
        assert result.exit_code == 0, result.stdout
        assert f'Upgraded user config from 0.0.1 to {version.pep440}' in result.stdout
        # Comments and formatting stay
        assert path.read_text(encoding='utf-8') == upgraded(OLD)

    def test_backup(self, cli, locs):
        path = locs.write(locs.local, OLD)
        result = cli('upgrade', writes=True)
        backup = path.with_name('xtl.toml.bak')
        assert backup.read_text(encoding='utf-8') == OLD
        assert f'Backup: {backup}' in result.stdout

    def test_no_backup(self, cli, locs):
        path = locs.write(locs.local, OLD)
        cli('upgrade', '--no-backup', writes=True)
        assert not path.with_name('xtl.toml.bak').exists()
        assert path.read_text(encoding='utf-8') == upgraded(OLD)

    def test_diff(self, cli, locs):
        path = locs.write(locs.local, OLD)
        result = cli('upgrade', '--diff')
        assert result.exit_code == 0
        assert '-version = "0.0.1" # mine' in result.stdout
        assert f'+version = "{version.pep440}" # mine' in result.stdout
        assert path.read_text(encoding='utf-8') == OLD

    @pytest.mark.parametrize('text, code', [(OLD, 1), (f'version = "{version.pep440}"\n', 0)], ids=['old', 'current'])
    def test_check(self, cli, locs, text, code):
        path = locs.write(locs.local, text)
        result = cli('upgrade', '--check')
        assert result.exit_code == code
        assert ('needs upgrading' in result.stdout) == bool(code)
        assert path.read_text(encoding='utf-8') == text

    def test_check_with_diff(self, cli, locs):
        locs.write(locs.local, OLD)
        result = cli('upgrade', '--check', '--diff')
        assert result.exit_code == 1
        assert '+version' in result.stdout

    def test_add_missing(self, cli, locs):
        # Also on a file that is up to date: every setting, unset ones commented out
        path = locs.write(locs.local, f'version = "{version.pep440}"\n[units]\ntemperature = "K" # kelvin\n')
        result = cli('upgrade', '--add-missing', '--no-backup', writes=True)
        assert result.exit_code == 0, result.stdout
        assert f'Updated local config ({version.pep440}): added the missing settings' in result.stdout
        text = path.read_text(encoding='utf-8')
        assert 'temperature = "K" # kelvin\n' in text
        assert '# path = ...  # Directory of binaries that every job finds' in text
        assert XTLSettings.from_toml(text).to_dict() == XTLSettings.from_toml('[units]\ntemperature = "K"\n').to_dict()
        # Nothing more to add the second time
        assert 'already up to date' in cli('upgrade', '--add-missing').stdout

    def test_current(self, cli, locs):
        text = f'# mine\nversion = "{version.pep440}"\n'
        path = locs.write(locs.local, text)
        result = cli('upgrade')
        assert result.exit_code == 0
        assert 'already up to date' in result.stdout
        assert path.read_text(encoding='utf-8') == text

    def test_clean(self, cli, locs):
        path = locs.write(locs.local, OLD)
        result = cli('upgrade', '--clean', writes=True)
        assert result.exit_code == 0, result.stdout
        assert path.read_text(encoding='utf-8') == XTLSettings.from_toml(OLD).to_toml(comments=True)

    def test_file(self, cli, locs):
        # Any file, also outside the lookup path; the active file is left alone
        active = locs.write(locs.local, OLD)
        path = locs.write(locs.root / 'other.cfg', OLD)
        result = cli('upgrade', str(path), writes=True)
        assert result.exit_code == 0, result.stdout
        assert path.read_text(encoding='utf-8') == upgraded(OLD)
        assert active.read_text(encoding='utf-8') == OLD

    def test_line_endings_stay(self, cli, locs):
        path = locs.root / 'crlf.toml'
        path.write_bytes(OLD.replace('\n', '\r\n').encode('utf-8'))
        result = cli('upgrade', str(path), writes=True)
        assert result.exit_code == 0, result.stdout
        assert path.read_bytes() == upgraded(OLD).replace('\n', '\r\n').encode('utf-8')

    @pytest.mark.parametrize(
        'text, message',
        [
            (INVALID, 'Error: units.temperature: '),
            ('[units\n', 'Error: '),
            ('version = "99.0.0"\n', 'cannot be downgraded'),
        ],
        ids=['invalid-values', 'invalid-toml', 'newer'],
    )
    def test_errors_leave_the_file(self, cli, locs, text, message):
        path = locs.write(locs.local, text)
        result = cli('upgrade')
        assert result.exit_code == 1
        assert message in result.stdout
        assert path.read_text(encoding='utf-8') == text

    @pytest.mark.parametrize('name', ['missing.toml', '.'])
    def test_bad_file(self, cli, locs, name):
        result = cli('upgrade', str(locs.root / name))
        assert result.exit_code == 2
