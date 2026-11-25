from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
import tomlkit

from xtl_core.options.toml import comment_text, toml_item, toml_scalar, with_comment
from xtl_core.system.os.permissions import FilePermissions


def dumps(value, comment=None) -> str:
    return tomlkit.dumps({'v': toml_item(value, comment)})


class TestTomlScalar:
    @pytest.mark.parametrize(
        'path', [Path('a', 'b'), PurePosixPath('a/b'), PureWindowsPath('a\\b')], ids=['Path', 'posix', 'windows']
    )
    def test_paths_posix(self, path):
        assert toml_scalar(path) == 'a/b'

    def test_file_permissions(self):
        assert toml_scalar(FilePermissions(0o755)) == '755'

    @pytest.mark.parametrize('value', [1, 'a', [1], None], ids=['int', 'str', 'list', 'None'])
    def test_others_unchanged(self, value):
        assert toml_scalar(value) is value


class TestCommentText:
    @pytest.mark.parametrize(
        'description, expected',
        [('one', 'one'), ('two\nlines', 'two lines'), ('  indented\n   lines  ', 'indented lines'), ('\n \n', '')],
        ids=['one', 'two', 'indented', 'blank'],
    )
    def test_one_line(self, description, expected):
        assert comment_text(description) == expected


class TestTomlItem:
    @pytest.mark.parametrize('name', ['a"b', 'a\\b', 'a\tb', 'a\x7fb', 'α β'], ids=['quote', 'bs', 'tab', 'del', 'utf'])
    def test_paths_escaped(self, name):
        """Regression: paths were written between quotes without escaping, so ``"`` gave
        invalid TOML and ``\\b`` turned into a backspace."""
        assert tomlkit.parse(dumps(PurePosixPath(name))).unwrap() == {'v': name}

    def test_file_permissions(self):
        assert dumps(FilePermissions(0o755)) == 'v = "755"\n'

    def test_known_types(self):
        assert dumps(date(2026, 11, 25)) == 'v = 2026-11-25\n'

    def test_unknown_types_as_str(self):
        class Thing:
            def __str__(self):
                return 'thing'

        assert dumps(Thing()) == 'v = "thing"\n'

    def test_comment(self):
        assert dumps(1, 'one') == 'v = 1 # one\n'

    @pytest.mark.parametrize('comment', [None, '', ' \n '], ids=['None', 'empty', 'blank'])
    def test_no_comment(self, comment):
        assert dumps(1, comment) == 'v = 1\n'


class TestWithComment:
    def test_table_header(self):
        table = tomlkit.table()
        table.add('a', 1)
        assert tomlkit.dumps({'t': with_comment(table, 'a\ntable')}) == '[t] # a table\na = 1\n'
