import typer

from .simulate import app as simulate_app

from xtl.cli.utilities import epilog


app = typer.Typer(
    name='xtl.nanobragg',
    help='nanoBragg simulations and utilities',
    add_completion=False,
    rich_markup_mode='rich',
    epilog=epilog
)
app.add_typer(simulate_app)
