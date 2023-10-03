import typer


class CliIO:

    def __init__(self, debug: bool = False, verbose: int = 0, silent: bool = False):
        self.indent = '\t'
        self.debug = debug
        self.verbose = verbose
        self.silent = silent
        self.echo_style = {
            'info': {},
            'debug': {'fg': typer.colors.BRIGHT_MAGENTA},
            'warning': {'fg': typer.colors.YELLOW},
            'error': {'fg': typer.colors.BRIGHT_RED}
        }

    def echo(self, message: str, level='info', verbose=0):
        style = self.echo_style.get(level, {})
        if self.debug:
            typer.secho(message, **style)
        else:
            if level == 'warning' and not self.silent:
                typer.secho(message, **style)
            elif level == 'error':
                typer.secho(message, **style)
            elif level == 'debug':
                pass
            elif self.verbose >= verbose and not self.silent:
                typer.secho(message, **style)
