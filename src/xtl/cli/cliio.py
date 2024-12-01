import traceback

import typer
import rich.box
import rich.console
import rich.pretty
import rich.prompt
import rich.table
import rich.text

from xtl.config import cfg


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

    def echo(self, message: str, level='info', verbose=0, **kwargs):
        nl = kwargs.get('nl', True)
        style = self.echo_style.get(level, {}) | kwargs.get('style', {})
        if self.debug:
            typer.secho(message, nl=nl, **style)
        else:
            if level == 'warning' and not self.silent:
                typer.secho(message, nl=nl, **style)
            elif level == 'error':
                typer.secho(message, nl=nl, **style)
            elif level == 'debug':
                pass
            elif self.verbose >= verbose and not self.silent:
                typer.secho(message, nl=nl, **style)


class Console(rich.console.Console):

    def __init__(self, verbose: int = 0, debug: bool = False, rich_output: bool = cfg['cli']['rich_output'].value,
                 striped_table_rows: bool = cfg['cli']['striped_table_rows'].value,
                 **console_kwargs):
        self.verbose = verbose
        self.debug = debug
        if isinstance(rich_output, str):
            self._rich_output = rich_output.lower() == 'true'
        else:
            self._rich_output = rich_output
        if isinstance(striped_table_rows, str):
            self._striped_table_rows = striped_table_rows.lower() == 'true'
        else:
            self._striped_table_rows = striped_table_rows

        super().__init__(**console_kwargs)

    def confirm(self, message: str, **kwargs):
        prompt = rich.prompt.Confirm(console=self)
        if not prompt.ask(message, **kwargs):
            raise typer.Abort()

    def print(self, *args, **kwargs):
        markup = kwargs.get('markup', None)
        if markup is None:
            markup = self._markup
        messages = []
        for arg in args:
            if isinstance(arg, str) and markup:
                messages.append(rich.text.Text.from_markup(arg))
            else:
                messages.append(arg)
        if not self._rich_output:
            # BUG: escaped characters that are not markup are also striped, e.g. \[text] will be lost
            messages = [m.plain for m in messages if isinstance(m, rich.text.Text)]
        super().print(*messages, **kwargs)

    def pprint(self, obj, **kwargs):
        rich.pretty.pprint(obj, **kwargs)

    def print_table(self, table: rich.table.Table | list, headers: list[str], column_kwargs: list[dict] = None,
                    table_kwargs: dict = None, **kwargs):
        # Default arguments
        if column_kwargs is None:
            column_kwargs = [{} for _ in headers]
        else:
            if len(column_kwargs) != len(headers):
                raise ValueError('Number of items in column_kwargs must match the number of items in headers')
        if table_kwargs is None:
            table_kwargs = {}

        # Create Table instance if not provided
        if not isinstance(table, rich.table.Table):
            t = rich.table.Table(**table_kwargs)
            for header, col_kwargs in zip(headers, column_kwargs):
                t.add_column(header, **col_kwargs)
            for row in table:
                t.add_row(*row)
            table = t

        if self._striped_table_rows:
            table.row_styles = ['', 'dim']
        if not self._rich_output:
            table.header_style = 'none'
            table.title_style = 'none'
            table.row_styles = None
            table.caption_style = 'none'
            for column in table.columns:
                column.style = None
        super().print(table, **kwargs)

    def print_traceback(self, exc: Exception, indent: str = ''):
        if self.debug:  # Format traceback with rich
            try:
                raise exc
            except Exception:
                super().print_exception(show_locals=True)
        elif self.verbose:  # Standard traceback
            for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
                self.print(f'{indent}{line}', style='red dim')
        else:  # Only print the exception
            self.print(f'{indent}{exc}', style='red dim')