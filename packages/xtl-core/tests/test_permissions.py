import dataclasses
import os

import pytest

from xtl_core.options import Option, Options
from xtl_core.options.serializers import PermissionOctal
from xtl_core.system import os as xtl_os
from xtl_core.system.compatibility import OS_POSIX, OS_WINDOWS
from xtl_core.system.os import permissions
from xtl_core.system.os.permissions import (
    FileType,
    FilePermissionsBit,
    FilePermissions,
    get_permissions_in_decimal,
    chmod_recursively,
)


posix_only = pytest.mark.skipif(not OS_POSIX, reason='POSIX modes')
not_as_root = pytest.mark.skipif(OS_POSIX and os.geteuid() == 0, reason='root ignores permissions')


def mode(path) -> int:
    return path.stat().st_mode & 0o777


@pytest.mark.parametrize('name', permissions.__all__)
def test_reexported_from_os(name):
    assert getattr(xtl_os, name) is getattr(permissions, name)
    assert name in xtl_os.__all__


class TestFileType:
    def test_values(self):
        assert [t.value for t in FileType] == ['-', 'd', 'l', 'c', 'b', 's', 'p']
        assert FileType('d') is FileType.DIRECTORY
        assert str(FileType.DIRECTORY) == 'd'


class TestFilePermissionsBit:
    @pytest.mark.parametrize('value', range(8))
    def test_int(self, value):
        assert FilePermissionsBit(value).value == value

    @pytest.mark.parametrize(
        'value, expected',
        [
            # Canonical
            ('---', 0o0),
            ('--x', 0o1),
            ('-w-', 0o2),
            ('-wx', 0o3),
            ('r--', 0o4),
            ('r-x', 0o5),
            ('rw-', 0o6),
            ('rwx', 0o7),
            # Short
            ('', 0o0),
            ('-', 0o0),
            ('x', 0o1),
            ('w', 0o2),
            ('wx', 0o3),
            ('r', 0o4),
            ('rx', 0o5),
            ('rw', 0o6),
            # Octal
            ('0o0', 0o0),
            ('0o5', 0o5),
            ('0o7', 0o7),
            # Nothing
            (None, 0o0),
        ],
    )
    def test_str(self, value, expected):
        assert FilePermissionsBit(value).value == expected

    @pytest.mark.parametrize('value', [-1, 8, 'abc', 'xr', 'wr', 'r-', '-w', 'rwxr', 'rrx', '0o8', '0o10', '7'])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            FilePermissionsBit(value)

    @pytest.mark.parametrize('value', [7.0, [7], b'rwx'])
    def test_invalid_type(self, value):
        with pytest.raises(TypeError, match='must be an int or str'):
            FilePermissionsBit(value)

    @pytest.mark.xfail(strict=True, reason='bool is accepted as an int')
    def test_bool(self):
        with pytest.raises(TypeError):
            FilePermissionsBit(True)

    @pytest.mark.parametrize(
        'value, short, canonical, as_tuple',
        [
            (0o0, '-', '---', (False, False, False)),
            (0o1, 'x', '--x', (False, False, True)),
            (0o2, 'w', '-w-', (False, True, False)),
            (0o3, 'wx', '-wx', (False, True, True)),
            (0o4, 'r', 'r--', (True, False, False)),
            (0o5, 'rx', 'r-x', (True, False, True)),
            (0o6, 'rw', 'rw-', (True, True, False)),
            (0o7, 'rwx', 'rwx', (True, True, True)),
        ],
    )
    def test_forms(self, value, short, canonical, as_tuple):
        b = FilePermissionsBit(value)
        assert b.string == short
        assert b.string_canonical == canonical
        assert b.octal == f'0o{value}'
        assert b.decimal == value
        assert b.tuple == as_tuple
        assert (b.can_read, b.can_write, b.can_execute) == as_tuple
        assert str(b) == short
        assert repr(b) == f'FilePermissionsBit(value={canonical!r})'

    def test_toggle(self):
        b = FilePermissionsBit('r-x')
        b.can_read = False
        assert b.string_canonical == '--x'
        b.can_write = True
        assert b.string_canonical == '-wx'
        b.can_execute = False
        assert b.string_canonical == '-w-'
        assert b.value == 0o2

    def test_set_unchanged(self):
        b = FilePermissionsBit('r-x')
        b.can_read = True
        b.can_write = False
        b.can_execute = True
        assert b.value == 0o5

    @pytest.mark.parametrize('name', ['can_read', 'can_write', 'can_execute'])
    def test_set_not_bool(self, name):
        b = FilePermissionsBit(0o5)
        with pytest.raises(TypeError, match='must be a bool'):
            setattr(b, name, 1)
        assert b.value == 0o5

    def test_compare(self):
        assert FilePermissionsBit('rx') == FilePermissionsBit(0o5) == FilePermissionsBit('r-x')
        assert FilePermissionsBit('rx') != FilePermissionsBit('rw')
        assert FilePermissionsBit(0o4) < FilePermissionsBit(0o5)
        assert max(FilePermissionsBit(0o6), FilePermissionsBit(0o7), FilePermissionsBit(0o1)).value == 0o7

    def test_dataclass(self):
        assert dataclasses.is_dataclass(FilePermissionsBit)
        assert [f.name for f in dataclasses.fields(FilePermissionsBit)] == ['value']

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(FilePermissionsBit(0o5))


class TestFilePermissions:
    @pytest.mark.parametrize(
        'value, expected',
        [
            (0o750, 0o750),
            (0o644, 0o644),
            (0o010, 0o010),
            (0o776, 0o776),
            ('750', 0o750),
            ('000', 0o000),
            ('0o750', 0o750),
            ('0o777', 0o777),
            ('rwxr-x---', 0o750),
            ('rwxr-xr-x', 0o755),
            ('---------', 0o000),
            ('-rwxr--r--', 0o744),
            ('drwxr-x---', 0o750),
            # A single part: the owner's
            ('rwx', 0o700),
            ('rx', 0o500),
        ],
    )
    def test_one_value(self, value, expected):
        assert FilePermissions(value).decimal == expected

    @pytest.mark.xfail(strict=True, reason='the range check excludes 0o777')
    def test_777(self):
        assert FilePermissions(0o777).decimal == 0o777

    @pytest.mark.xfail(strict=True, reason='a lone int up to 7 is read as the owner digit')
    @pytest.mark.parametrize('value', [0o1, 0o5, 0o7])
    def test_lone_small_int(self, value):
        assert FilePermissions(value).decimal == value

    @pytest.mark.xfail(strict=True, reason='4-digit octal strings are rejected')
    def test_leading_zero(self):
        assert FilePermissions('0755').decimal == 0o755

    @pytest.mark.parametrize('value', [0o1000, -0o10, '999', 'rwxr-x', 'rwxr-x---x', 'rwzr-x---', 'abc'])
    def test_one_value_invalid(self, value):
        with pytest.raises(ValueError):
            FilePermissions(value)

    @pytest.mark.parametrize(
        'parts, expected',
        [
            ((6, 4, 4), 0o644),
            ((5, 0, 0), 0o500),
            (('rwx', '', None), 0o700),
            (('', 'rwx', 3), 0o073),
            (('rw-', 'r', '0o4'), 0o644),
            ((FilePermissionsBit(7), FilePermissionsBit('rx'), 'x'), 0o751),
        ],
    )
    def test_parts(self, parts, expected):
        assert FilePermissions(*parts).decimal == expected

    def test_keywords(self):
        assert FilePermissions(owner=7, group='r-x').decimal == 0o750
        assert FilePermissions(other='rwx').decimal == 0o007
        assert FilePermissions(owner='rw', group='r', other='r', file_type='-').string == '-rw-r--r--'

    def test_defaults(self):
        p = FilePermissions()
        assert p.decimal == 0o000
        assert p.file_type is None
        assert p.owner == p.group == p.other == FilePermissionsBit(0)

    def test_parts_invalid(self):
        with pytest.raises(ValueError, match="'group' must be a valid permission string"):
            FilePermissions(7, 'abc', 0)
        with pytest.raises(ValueError, match="'other' must be a valid permission string"):
            FilePermissions(7, 5, 8)
        with pytest.raises(TypeError, match="'group' must be an int, str or FilePermissionsBit, not float"):
            FilePermissions(7, 5.0, 0)

    def test_from_permissions(self):
        p = FilePermissions('drwxr-x---')
        q = FilePermissions(p)
        assert q == p
        assert q.string == 'drwxr-x---'
        assert q.file_type is FileType.DIRECTORY

    @pytest.mark.parametrize(
        'file_type, expected', [('d', FileType.DIRECTORY), ('-', FileType.FILE), (FileType.FIFO, FileType.FIFO)]
    )
    def test_file_type(self, file_type, expected):
        p = FilePermissions(0o750, file_type=file_type)
        assert p.file_type is expected
        assert p.string == f'{expected.value}rwxr-x---'

    def test_file_type_from_string(self):
        assert FilePermissions('-rw-r-----').file_type is FileType.FILE
        assert FilePermissions('lrwxrwxrwx').file_type is FileType.SYMLINK
        assert FilePermissions('rw-r-----').file_type is None

    def test_file_type_invalid(self):
        with pytest.raises(TypeError):
            FilePermissions(0o750, file_type='z')
        with pytest.raises(TypeError):
            FilePermissions(0o750, file_type=4)

    @pytest.mark.xfail(strict=True, reason="the error names the enum's metaclass")
    def test_file_type_invalid_message(self):
        with pytest.raises(TypeError, match='FileType'):
            FilePermissions(0o750, file_type=4)

    @pytest.mark.parametrize(
        'value, octal, string',
        [
            (0o750, '0o750', 'rwxr-x---'),
            (0o644, '0o644', 'rw-r--r--'),
            ('000', '0o000', '---------'),
            ('0o777', '0o777', 'rwxrwxrwx'),
            (0o010, '0o010', '-----x---'),
        ],
    )
    def test_forms(self, value, octal, string):
        p = FilePermissions(value)
        assert p.octal == octal
        assert p.decimal == int(octal, 8)
        assert p.string == string
        assert str(p) == string
        assert repr(p) == f'FilePermissions(value={string!r})'

    def test_parts_as_bits(self):
        p = FilePermissions(0o751)
        assert p.owner == FilePermissionsBit(7)
        assert p.group == FilePermissionsBit(5)
        assert p.other == FilePermissionsBit(1)
        assert p.tuple == ((True, True, True), (True, False, True), (False, False, True))

    def test_change_part(self):
        p = FilePermissions(0o750)
        p.owner.can_write = False
        p.other.can_read = True
        assert p.decimal == 0o554
        assert p.string == 'r-xr-xr--'

    def test_compare(self):
        assert FilePermissions('750') == FilePermissions(0o750) == FilePermissions('rwxr-x---')
        assert FilePermissions('750') != FilePermissions('755')
        # The file type is left out
        assert FilePermissions('drwxr-x---') == FilePermissions('-rwxr-x---') == FilePermissions('rwxr-x---')
        # By mode
        assert FilePermissions(0o640) < FilePermissions(0o700)
        assert FilePermissions(0o077) < FilePermissions(0o700)
        assert FilePermissions(0o750) < FilePermissions(0o751)

    def test_dataclass(self):
        assert dataclasses.is_dataclass(FilePermissions)
        assert [f.name for f in dataclasses.fields(FilePermissions)] == ['owner', 'group', 'other', 'file_type']

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(FilePermissions(0o750))

    @pytest.mark.parametrize('value', ['750', '0o750', 'rwxr-x---', 'drwxr-x---'])
    def test_from_string(self, value):
        assert FilePermissions.from_string(value) == FilePermissions(value)
        assert FilePermissions.from_string(value).string == FilePermissions(value).string


class TestFromPath:
    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FilePermissions.from_path(tmp_path / 'missing')

    @posix_only
    @pytest.mark.parametrize(
        'value',
        [
            0o640,
            0o600,
            0o755,
            *(
                pytest.param(v, marks=pytest.mark.xfail(strict=True, reason='the mode is split by its octal digits'))
                for v in (0o000, 0o077, 0o055)
            ),
        ],
    )
    def test_file(self, tmp_path, value):
        f = tmp_path / 'file'
        f.touch()
        f.chmod(value)
        p = FilePermissions.from_path(f)
        assert p.decimal == value
        assert p.file_type is FileType.FILE

    @posix_only
    def test_directory(self, tmp_path):
        d = tmp_path / 'dir'
        d.mkdir()
        d.chmod(0o750)
        p = FilePermissions.from_path(str(d))
        assert p.string == 'drwxr-x---'

    @posix_only
    def test_symlink_followed(self, tmp_path):
        f = tmp_path / 'file'
        f.touch()
        f.chmod(0o640)
        link = tmp_path / 'link'
        link.symlink_to(f)
        p = FilePermissions.from_path(link)
        assert p.decimal == 0o640
        assert p.file_type is FileType.FILE

    @posix_only
    def test_fifo(self, tmp_path):
        fifo = tmp_path / 'fifo'
        os.mkfifo(fifo, 0o600)
        p = FilePermissions.from_path(fifo)
        assert p.file_type is FileType.FIFO

    @pytest.mark.skipif(not OS_WINDOWS, reason='Windows modes')
    def test_windows(self, tmp_path):
        f = tmp_path / 'file'
        f.touch()
        assert FilePermissions.from_path(f).string == '-rw-rw-rw-'
        assert FilePermissions.from_path(tmp_path).string == 'drwxrwxrwx'


class TestGetPermissionsInDecimal:
    @pytest.mark.parametrize(
        'value, expected',
        [('0o755', 0o755), ('711', 0o711), (0o640, 0o640), ('rw-r-----', 0o640), (FilePermissions(0o700), 0o700)],
    )
    def test_valid(self, value, expected):
        assert get_permissions_in_decimal(value) == expected

    @pytest.mark.parametrize('value', [777, 999, 'asdf'])  # base 10 integers are out of range
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            get_permissions_in_decimal(value)

    @pytest.mark.parametrize('value', [755.0, None])
    def test_invalid_type(self, value):
        with pytest.raises(TypeError):
            get_permissions_in_decimal(value)

    @pytest.mark.xfail(strict=True, reason="the error names `type`, FilePermissions' metaclass")
    def test_invalid_type_message(self):
        with pytest.raises(TypeError, match='FilePermissions'):
            get_permissions_in_decimal(755.0)


class TestChmodRecursively:
    @pytest.fixture
    def tree(self, tmp_path):
        """
        ``root/{a.txt, sub/{b.txt, deeper/c.txt}}`` and a symlink ``root/link`` to a file
        outside the tree (POSIX only).
        """
        root = tmp_path / 'root'
        (root / 'sub' / 'deeper').mkdir(parents=True)
        for f in ('a.txt', 'sub/b.txt', 'sub/deeper/c.txt'):
            (root / f).touch()
        if OS_POSIX:
            outside = tmp_path / 'outside.txt'
            outside.touch()
            outside.chmod(0o600)
            (root / 'link').symlink_to(outside)
        return root

    @staticmethod
    def files(root):
        return [root / 'a.txt', root / 'sub' / 'b.txt', root / 'sub' / 'deeper' / 'c.txt']

    @staticmethod
    def directories(root):
        return [root, root / 'sub', root / 'sub' / 'deeper']

    def test_nothing_to_set(self, tree):
        with pytest.raises(ValueError, match='At least one of'):
            chmod_recursively(tree)

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            chmod_recursively(tmp_path / 'missing', files_permissions='640', directories_permissions='750')

    @posix_only
    @pytest.mark.parametrize(
        'before, after',
        [((0o600, 0o700), (0o644, 0o755)), ((0o644, 0o755), (0o600, 0o700))],
        ids=['more', 'less'],
    )
    def test_tree(self, tree, before, after):
        for f in self.files(tree):
            f.chmod(before[0])
        for d in self.directories(tree):
            d.chmod(before[1])
        chmod_recursively(tree, files_permissions=after[0], directories_permissions=FilePermissions(after[1]))
        assert [mode(f) for f in self.files(tree)] == [after[0]] * 3
        assert [mode(d) for d in self.directories(tree)] == [after[1]] * 3
        # Symbolic links are skipped, and so are their targets
        assert mode(tree.parent / 'outside.txt') == 0o600

    @posix_only
    def test_symlink_root(self, tree):
        sub = tree / 'sub'
        sub.chmod(0o700)
        link = tree.parent / 'link_to_sub'
        link.symlink_to(sub)
        chmod_recursively(link, files_permissions='600', directories_permissions='755')
        assert mode(sub) == 0o700

    @posix_only
    def test_file(self, tree):
        f = tree / 'a.txt'
        f.chmod(0o600)
        chmod_recursively(f, files_permissions='644', directories_permissions='755')
        assert mode(f) == 0o644

    @posix_only
    @pytest.mark.xfail(strict=True, reason='the missing permissions are converted too')
    def test_files_only(self, tree):
        for d in self.directories(tree):
            d.chmod(0o750)
        chmod_recursively(tree, files_permissions='640')
        assert [mode(f) for f in self.files(tree)] == [0o640] * 3
        assert [mode(d) for d in self.directories(tree)] == [0o750] * 3

    @posix_only
    @not_as_root
    @pytest.mark.xfail(strict=True, reason='"more permissive" compares the modes as numbers')
    def test_order_by_bits(self, tree):
        # 0o600 > 0o500 as a number, but it takes the execute bit away: changing the root
        #  first would leave its contents unreachable
        for d in self.directories(tree):
            d.chmod(0o500)
        try:
            chmod_recursively(tree, files_permissions='600', directories_permissions='600')
        finally:
            for d in self.directories(tree):
                d.chmod(0o700)
        assert [mode(f) for f in self.files(tree)] == [0o600] * 3


class _WithPermissions(Options):
    permissions: FilePermissions = Option(
        default=FilePermissions(0o700), cast_as=FilePermissions, formatter=PermissionOctal
    )


class TestInOptions:
    def test_default_copied(self):
        a, b = _WithPermissions(), _WithPermissions()
        assert a.permissions == FilePermissions(0o700)
        assert a.permissions is not b.permissions

    @pytest.mark.parametrize('value', ['750', 0o750, 'rwxr-x---', FilePermissions(0o750)])
    def test_cast(self, value):
        assert _WithPermissions(permissions=value).permissions == FilePermissions(0o750)

    def test_invalid(self):
        with pytest.raises(ValueError):
            _WithPermissions(permissions='abc')

    def test_toml(self):
        o = _WithPermissions(permissions='750')
        s = o.to_toml()
        assert 'permissions = "750"' in s
        assert _WithPermissions.from_toml(s) == o

    def test_json(self):
        o = _WithPermissions(permissions='750')
        assert _WithPermissions.from_json(o.to_json()) == o

    def test_serializer(self):
        assert PermissionOctal(FilePermissions(0o750)) == '750'
        assert PermissionOctal(FilePermissions('000')) == '000'
