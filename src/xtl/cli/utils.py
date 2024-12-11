import asyncio
from functools import wraps

import click
import tabulate



def typer_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


# Tabulate formatting settings
tabulate.MIN_PADDING = 0
tabulate._table_formats['two_columns'] = tabulate.TableFormat(
    lineabove=None,
    linebelowheader=tabulate.Line("", "-", "  ", ""),
    linebetweenrows=None,
    linebelow=None,
    headerrow=tabulate.DataRow("", "   ", ""),
    datarow=tabulate.DataRow("", " : ", ""),
    padding=0,
    with_header_hide=None,
)


def update_docstring(field, value):
    """
    Decorator that replaces a field in a function's docstring with a specified value.

    :param field: String to replace
    :param value: Value to replace string with
    :return:
    """
    def _doc(func):
        func.__doc__ = func.__doc__.replace(field, value)
        return func
    return _doc


def dict_to_table(dict):
    """
    Convert a dictionary to a list of lists.

    :param dict dict: Dictionary to convert
    :return: List of lists
    :rtype: list
    """
    table = []
    for key, value in dict.items():
        table.append([key, value])
    return table


def flatten_dict(dict_):
    new_dict = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    new_dict[f"{key}.{k}"] = flatten_dict(v)
                else:
                    new_dict[f"{key}.{k}"] = v
        else:
            new_dict[key] = value
    return new_dict


class OutputQueue:

    def __init__(self, debug=False):
        self.debug = debug
        self._echo_queue = []

    def append_to_queue(self, entry):
        self._echo_queue.append(entry)
        if self.debug:
            self.print()

    def pop_from_queue(self):
        return self._echo_queue.pop(-1)

    def append_table(self, table, title: str = "", *tabulate_args, **tabulate_kwargs):
        """

        :param table: Iterable to be passed to tabulate
        :param str title:
        :param tabulate_args: Extra tabulate args
        :param tabulate_kwargs: Extra tabulate kwargs
        :return:
        """
        if title:
            self.append_to_queue((f'{title.upper()}\n{"="*len(title)}', {'bold': True}))
        self.append_to_queue((tabulate.tabulate(table, *tabulate_args, **tabulate_kwargs), {}))
        self.append_to_queue(('\n', {}))

    def _output_generator(self):
        while len(self._echo_queue) > 0:
            yield self._echo_queue.pop(0)

    def print(self):
        entries = self._output_generator()
        for entry in entries:
            content, instructions = entry
            click.secho(content, **instructions)


class GroupWithCommandOptions(click.Group):
    """
    Allow application of options to group with multi command.

    Code from https://stackoverflow.com/a/48509916
    """

    def add_command(self, cmd, name=None):
        click.Group.add_command(self, cmd, name=name)

        # Add the group parameters to the command
        for param in self.params:
            cmd.params.append(param)

        # Hook the commands invoke with our own
        cmd.invoke = self.build_command_invoke(cmd.invoke)
        self.invoke_without_command = True

    def build_command_invoke(self, original_invoke):

        def command_invoke(ctx):
            """
            Insert invocation of group function
            """

            # Separate the group parameters
            ctx.obj = dict(_params=dict())
            for param in self.params:
                name = param.name
                ctx.obj['_params'][name] = ctx.params[name]
                del ctx.params[name]

            # Call the group function with its parameters
            params = ctx.params
            ctx.params = ctx.obj['_params']
            self.invoke(ctx)
            ctx.params = params

            # Now call the original invoke (the command)
            original_invoke(ctx)

        return command_invoke
