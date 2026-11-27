"""
Sphinx directives for tables generated from XTL's settings, so that the docs follow the code.

The shells XTL writes batch files for, with their executables, and a note for each (a shell without one is a
warning)::

    .. shells::
       :bash: The default on Linux and macOS.
       :cmd: The default on Windows.

The keys of a dependency's table in ``xtl.toml``, with their descriptions::

    .. dependency-keys::

The known dependencies, the programs XTL looks for, and the commands that need them (those that declare them with
``@requires``)::

    .. dependency-programs::
"""

from collections import defaultdict
import re

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)


def _commands_by_dependency() -> dict[str, list[str]]:
    """
    The commands of XTL's entry points that declare each dependency, e.g. ``{'atsas': ['xtl.saxs compare']}``.
    """
    from importlib.metadata import entry_points

    import typer

    used_by: dict[str, list[str]] = defaultdict(list)

    def walk(app: typer.Typer, prefix: str) -> None:
        for command in app.registered_commands:
            name = command.name or command.callback.__name__.replace('_', '-')
            for dependency in getattr(command.callback, 'dependencies', ()):
                used_by[dependency].append(f'{prefix} {name}')
        for group in app.registered_groups:
            # Groups without a name add their commands to the parent
            walk(group.typer_instance, f'{prefix} {group.name}' if group.name else prefix)

    for entry_point in entry_points(group='console_scripts'):
        if entry_point.dist is None or entry_point.dist.name != 'xtl':
            continue
        try:
            app = entry_point.load()
        except Exception as e:
            logger.warning('Cannot load the command %s: %s', entry_point.name, e)
            continue
        if isinstance(app, typer.Typer):
            walk(app, entry_point.name)
    return {dependency: sorted(commands) for dependency, commands in used_by.items()}


def _link(cls: type) -> str:
    """
    The dependency's name with its link, from the first line of its settings' docstring, e.g.
    ``Settings for `ATSAS <https://...>`_`` → ``ATSAS <https://...>`_``.
    """
    doc = (cls.__doc__ or '').strip()
    return re.sub(r'^Settings for\s+', '', doc.splitlines()[0]) if doc else cls.__name__


class _TableDirective(SphinxDirective):
    """
    A ``list-table``, from a header and rows of reST text.
    """

    widths: str = ''

    def rows(self) -> tuple[list[str], list[list[str]]]:
        raise NotImplementedError

    def run(self) -> list[nodes.Node]:
        header, rows = self.rows()
        lines = ['.. list-table::', '   :header-rows: 1', f'   :widths: {self.widths}', '']
        for row in [header, *rows]:
            lines.append(f'   * - {row[0]}')
            lines += [f'     - {cell}' for cell in row[1:]]
        container = nodes.container()
        self.state.nested_parse(StringList(lines, source=self.get_source_info()[0]), self.content_offset, container)
        return container.children


def _shells() -> list:
    """
    The shells XTL writes batch files for, as instances, without the system's default.
    """
    from xtl.jobs.shells import Shell

    return [member.get() for member in Shell if member != Shell.DEFAULT]


class ShellsDirective(_TableDirective):
    """
    The shells XTL writes batch files for, with their executables; a note for each shell by its name, e.g.
    ``:bash: The default on Linux and macOS.``
    """

    widths = '15 35 50'
    option_spec = {shell.name: directives.unchanged for shell in _shells()}

    def rows(self) -> tuple[list[str], list[list[str]]]:
        rows = [
            [f'``{shell.name}``', f'``{shell.executable}``', self.options.get(shell.name, '')] for shell in _shells()
        ]
        for shell in _shells():
            if shell.name not in self.options:
                logger.warning('No note for the shell %s', shell.name, location=self.get_location())
        return ['Shell', 'Executable', 'Notes'], rows


class DependencyKeysDirective(_TableDirective):
    """
    The keys of a dependency's table in ``xtl.toml``, with their descriptions.
    """

    widths = '20 80'

    def rows(self) -> tuple[list[str], list[list[str]]]:
        from xtl.config.settings import DependencySettings

        rows = [
            [f'``{name}``', field.description or '']
            for name, field in DependencySettings.model_fields.items()
            if not field.exclude
        ]
        return ['Key', 'Description'], rows


class DependencyProgramsDirective(_TableDirective):
    """
    The known dependencies, the programs XTL looks for, and the commands that need them.
    """

    widths = '20 30 50'

    def rows(self) -> tuple[list[str], list[list[str]]]:
        from xtl import settings

        used_by = _commands_by_dependency()
        rows = []
        for name, dependency in settings.dependencies.items():
            programs = ', '.join(f'``{program}``' for program in sorted(dependency.provides, key=str.lower))
            commands = ', '.join(f'``{command}``' for command in used_by.get(name, []))
            rows.append([f'``{name}``', programs, _link(type(dependency)) + (f': {commands}' if commands else '')])
        return ['Dependency', 'Programs', 'Used by'], rows


def setup(app):
    app.add_directive('shells', ShellsDirective)
    app.add_directive('dependency-keys', DependencyKeysDirective)
    app.add_directive('dependency-programs', DependencyProgramsDirective)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
