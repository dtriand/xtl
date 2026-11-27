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
