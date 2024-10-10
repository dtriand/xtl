from pathlib import Path
import stat

from xtl.automate.sites import ComputeSite, LocalSite

class BatchFile:
    HEADER = '#!/bin/bash'
    COMMENT_CHAR = '#'
    NEW_LINE_CHAR = '\n'
    FILE_EXT = '.sh'

    def __init__(self, name, filename, compute_site: ComputeSite = None):
        self._name = name
        self._filename = Path(filename).with_suffix(self.FILE_EXT)
        self._lines = []
        if compute_site is None:
            compute_site = LocalSite()
        elif not isinstance(compute_site, ComputeSite):
            raise TypeError(f"compute_site must be an instance of ComputeSite, not {type(compute_site)}")
        self._compute_site = compute_site

    def _add_line(self, line):
        self._lines.append(str(line) + self.NEW_LINE_CHAR)

    def add_command(self, command: str = ''):
        self._add_line(command)

    def add_commands(self, *commands):
        for command in commands:
            self.add_command(command)

    def add_comment(self, comment):
        self._add_line(f"{self.COMMENT_CHAR} {comment}")

    def load_modules(self, modules: str | list[str]):
        self.add_command(self._compute_site.load_modules(modules))

    def purge_modules(self):
        self.add_command(self._compute_site.purge_modules())

    def assign_variable(self, var_name, expression):
        self.add_command(str(var_name) + '=' + str(expression))

    def save(self, do_chmod=True):
        # Delete the file if it already exists
        self._filename.unlink(missing_ok=True)

        # Write contents to file
        text = self.HEADER + self.NEW_LINE_CHAR + ''.join(self._lines)
        self._filename.write_text(text, encoding='utf-8')

        # Update permissions (user: read, write, execute; group: read, write)
        if do_chmod:
            mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IWGRP  # rwxrw---- / 760
            self._filename.chmod(mode)