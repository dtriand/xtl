import typer

from .qq import app as qq_app


app = typer.Typer(name='correlate', help='Calculate intensity cross-correlation functions')
app.add_typer(qq_app)
