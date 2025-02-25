import typer

from .integrate_1d import app as integrate_1d_app


app = typer.Typer(name='integrate', help='Perform azimuthal integrations of X-ray data')
app.add_typer(integrate_1d_app)
