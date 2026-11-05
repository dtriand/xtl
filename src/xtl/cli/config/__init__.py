import typer

from xtl.cli.utilities import epilog
from xtl.cli.config.create import app as create_app
from xtl.cli.config.show import app as show_app
from xtl.cli.config.validate import app as validate_app
from xtl.cli.config.where import app as where_app

app = typer.Typer(
    name='xtl.config',
    help='Utilities for managing XTL configuration files',
    add_completion=False,
    rich_markup_mode='rich',
    no_args_is_help=True,
    epilog=epilog,
)
app.add_typer(create_app)
app.add_typer(validate_app)
app.add_typer(where_app)
app.add_typer(show_app)
