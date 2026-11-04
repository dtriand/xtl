import typer

from xtl.cli.utilities import epilog
from xtl.cli.config.where import app as where_app

app = typer.Typer(
    name='xtl.config',
    help='Utilities for managing XTL configuration files',
    add_completion=False,
    rich_markup_mode='rich',
    epilog=epilog
)
app.add_typer(where_app)
