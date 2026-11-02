import typer

from .compare import app as compare_app
from .subtract import app as subtract_app
from xtl.cli.utilities import epilog


app = typer.Typer(
    name='xtl.saxs',
    help='Utilities for scattering data',
    add_completion=False,
    rich_markup_mode='rich',
    epilog=epilog
)
app.add_typer(compare_app)
app.add_typer(subtract_app)
