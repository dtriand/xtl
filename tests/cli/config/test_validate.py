import pytest

from tests.cli.config.conftest import INVALID, VALID, WARNINGS


class TestValidate:
    def test_no_active(self, cli):
        result = cli('validate')
        assert result.exit_code == 0
        assert 'nothing to validate' in result.stdout

    def test_valid(self, cli, locs):
        locs.write(locs.user, VALID)
        result = cli('validate')
        assert result.exit_code == 0
        assert 'Validating user config' in result.stdout
        assert 'File is valid.' in result.stdout

    def test_warnings(self, cli, locs):
        locs.write(locs.local, WARNINGS)
        result = cli('validate')
        assert result.exit_code == 0
        assert result.stdout.count('Warning:') == 2
        assert 'units.colour' in result.stdout
        assert 'File is valid with 2 warnings.' in result.stdout

    def test_errors(self, cli, locs):
        locs.write(locs.local, INVALID)
        result = cli('validate')
        assert result.exit_code == 1
        assert 'Error: units.temperature: ' in result.stdout
        assert 'File is invalid' in result.stdout

    def test_file(self, cli, locs):
        # Any file, also outside the lookup path; the active file is not checked
        locs.write(locs.local, INVALID)
        path = locs.write(locs.root / 'other.cfg', VALID)
        result = cli('validate', str(path))
        assert result.exit_code == 0
        assert 'File is valid.' in result.stdout

    def test_file_invalid_toml(self, cli, locs):
        path = locs.write(locs.root / 'other.toml', '[units\n')
        result = cli('validate', str(path))
        assert result.exit_code == 1
        assert result.stdout.count('Error:') == 1

    @pytest.mark.parametrize('name', ['missing.toml', '.'])
    def test_bad_file(self, cli, locs, name):
        result = cli('validate', str(locs.root / name))
        assert result.exit_code == 2
