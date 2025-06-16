import typer

from .compare import app as compare_app


app = typer.Typer(name='xtl.saxs', help='Utilities for scattering data')
app.add_typer(compare_app)
