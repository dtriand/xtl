from pathlib import Path

import pytest

from xtl.common.environment import EnvVar, XTL_CONFIG


NAME = 'XTL_TEST_ENVVAR'


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    return monkeypatch


class TestEnvVar:
    def test_unset(self, env):
        var = EnvVar(NAME)
        assert var.raw is None
        assert not var.is_set
        assert var.value is None

    def test_set(self, env):
        env.setenv(NAME, 'foo')
        var = EnvVar(NAME)
        assert var.raw == 'foo'
        assert var.is_set
        assert var.value == 'foo'

    def test_empty_is_unset(self, env):
        env.setenv(NAME, '')
        var = EnvVar(NAME, default='bar')
        assert var.raw is None
        assert not var.is_set
        assert var.value == 'bar'

    def test_default(self, env):
        var = EnvVar(NAME, parser=int, default=3)
        assert var.value == 3
        env.setenv(NAME, '5')
        assert var.value == 5

    def test_parse_error_names_the_variable(self, env):
        env.setenv(NAME, 'five')
        with pytest.raises(ValueError, match=f"{NAME}: 'five'"):
            _ = EnvVar(NAME, parser=int).value

    def test_read_on_every_access(self, env):
        # Changes after the EnvVar is created are picked up
        var = EnvVar(NAME)
        assert var.value is None
        env.setenv(NAME, 'foo')
        assert var.value == 'foo'
        env.delenv(NAME)
        assert var.value is None

    def test_path_parser(self, monkeypatch, tmp_path):
        monkeypatch.setenv('XTL_TEST_DIR', str(tmp_path))
        monkeypatch.setenv('XTL_CONFIG', '$XTL_TEST_DIR/xtl.toml')
        assert XTL_CONFIG.value == tmp_path / 'xtl.toml'
        monkeypatch.setenv('XTL_CONFIG', '~/xtl.toml')
        assert XTL_CONFIG.value == Path.home() / 'xtl.toml'
