"""
Sphinx directives for shell commands that differ between shells, shown as tabs (with
`sphinx-design`). Tabs in the `shell` sync group switch together across the page, so a
reader picks their shell once.

Set an environment variable::

    .. set-env:: XTL_CONFIG /path/to/xtl.toml
       :windows: C:\\path\\to\\xtl.toml

For other commands, use a `tab-set` with the same sync group and keys::

    .. tab-set::
       :sync-group: shell

       .. tab-item:: Bash
          :sync: bash
       .. tab-item:: PowerShell
          :sync: pwsh
       .. tab-item:: cmd
          :sync: cmd
"""

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

# (label, sync key, lexer, prompt, command template, uses the Windows value)
SHELLS = [
    ('Bash', 'bash', 'console', '$ ', 'export {name}={value}', False),
    ('PowerShell', 'pwsh', 'ps1con', 'PS> ', '$env:{name} = "{value}"', True),
    ('cmd', 'cmd', 'doscon', '> ', 'set {name}={value}', True),
]


class SetEnvDirective(SphinxDirective):
    """
    Tabs with the command that sets an environment variable in each shell, for the
    current session. The ``:windows:`` option gives a different value for the Windows
    shells (*e.g.* a Windows path).
    """

    required_arguments = 2
    final_argument_whitespace = True
    option_spec = {'windows': directives.unchanged}

    def run(self) -> list[nodes.Node]:
        name, value = self.arguments
        windows = self.options.get('windows', value)

        lines = ['.. tab-set::', '   :sync-group: shell', '']
        for label, key, lexer, prompt, template, is_windows in SHELLS:
            command = template.format(name=name, value=windows if is_windows else value)
            lines += [
                f'   .. tab-item:: {label}',
                f'      :sync: {key}',
                '',
                f'      .. code-block:: {lexer}',
                '',
                f'         {prompt}{command}',
                '',
            ]

        container = nodes.container()
        self.state.nested_parse(StringList(lines, source=self.get_source_info()[0]), self.content_offset, container)
        return container.children


def setup(app):
    app.setup_extension('sphinx_design')
    app.add_directive('set-env', SetEnvDirective)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
