import typer

from .geometry import app as geometry_app


app = typer.Typer(name='xtl.diffraction', help='Utilities for diffraction data')
app.add_typer(geometry_app)
