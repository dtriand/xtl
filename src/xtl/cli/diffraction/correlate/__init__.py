import typer

from .qq import app as qq_app
from .fft import app as fft_app
from xtl.cli.cliio import epilog


app = typer.Typer(name='correlate', help='Calculate intensity cross-correlation functions',
                  add_completion=False, rich_markup_mode='rich', epilog=epilog)
app.add_typer(qq_app)
app.add_typer(fft_app)
