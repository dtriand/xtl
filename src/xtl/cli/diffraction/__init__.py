import typer

from .geometry import app as geometry_app
from .plot import app as plot_app
# from .sum import app as sum_app
from .integrate import app as integrate_app
from .correlate import app as correlate_app
# from .array import app as array_app


app = typer.Typer(name='xtl.diffraction', help='Utilities for diffraction data')
app.add_typer(geometry_app)
app.add_typer(plot_app)
# app.add_typer(sum_app)
app.add_typer(integrate_app)
app.add_typer(correlate_app)
# app.add_typer(array_app)
