import typer

from .plot_2d import app as plot_2d_app
from xtl.cli.cliio import epilog


app = typer.Typer(name='plot', help='Plot diffraction data',
                  add_completion=False, rich_markup_mode='rich', epilog=epilog)
app.add_typer(plot_2d_app)
