import typer

from .integrate_1d import app as integrate_1d_app
from xtl.cli.cliio import epilog


app = typer.Typer(name='integrate', help='Perform azimuthal integrations of X-ray data',
                  add_completion=False, rich_markup_mode='rich', epilog=epilog)
app.add_typer(integrate_1d_app)
