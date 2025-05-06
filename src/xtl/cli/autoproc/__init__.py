import typer

from .options import app as options_app
from .process import app as process_app
from .process_wf import app as process_wf_app
from .fixnml import app as fixnml_app
from .summarize import app as summarize_app

from xtl.cli.cliio import epilog


app = typer.Typer(name='xtl.autoproc', help='Execute multiple autoPROC jobs',
                  add_completion=False, rich_markup_mode='rich', epilog=epilog)
app.add_typer(options_app)
app.add_typer(process_app)
app.add_typer(process_wf_app)
app.add_typer(fixnml_app)
app.add_typer(summarize_app)
