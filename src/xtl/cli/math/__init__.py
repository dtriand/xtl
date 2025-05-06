import typer

from .spacing import app as spacing_app

from xtl.cli.cliio import epilog


app = typer.Typer(name='xtl.math', help='Various crystallographic calculators',
                  add_completion=False, rich_markup_mode='rich', epilog=epilog)
app.add_typer(spacing_app)
