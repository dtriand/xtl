import pytest
import toml
from typer.testing import CliRunner


# Sample config files
VALID = '[units]\ntemperature = "K"\n'
WARNINGS = 'version = "0.0.1"\n[units]\ncolour = "blue"\n'  # 2 warnings
INVALID = '[units]\ntemperature = "X"\n'


def defaults() -> dict:
    """The default settings, as TOML data."""
    from xtl.config.settings import XTLSettings

    return toml.loads(XTLSettings().to_toml())


@pytest.fixture
def cli(locs):
    """
    Run ``xtl.config``. Read-only runs (``writes=False``, the default) also check that
    nothing was written in any of the config locations.
    """
    from xtl.cli.config import app

    runner = CliRunner()

    def run(*args: str, writes: bool = False):
        before = locs.files()
        # Wide enough for the paths in the `where` table
        result = runner.invoke(app, list(args), env={'COLUMNS': '500'})
        if not writes:
            assert locs.files() == before
        return result

    return run
