import pytest

from xtl_core.system import os as xtl_os
from xtl_core.system.os import permissions
from xtl_core.system.os.permissions import get_permissions_in_decimal, FilePermissionsBit, FilePermissions


@pytest.mark.parametrize('name', permissions.__all__)
def test_reexported_from_os(name):
    assert getattr(xtl_os, name) is getattr(permissions, name)
    assert name in xtl_os.__all__


class TestPermissions:
    @pytest.mark.parametrize(
        'value,   expected',
        [
            ('0o755', 493),
            pytest.param(777, None, marks=pytest.mark.xfail(raises=ValueError)),  # base 10 integer
            ('711', 457),
            pytest.param(755.0, None, marks=pytest.mark.xfail(raises=TypeError)),
            pytest.param('asdf', None, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(999, None, marks=pytest.mark.xfail(raises=ValueError)),
        ],
    )
    def test_valid_permissions(self, value, expected):
        assert get_permissions_in_decimal(value) == expected


class TestFilePermissionsBit:
    @pytest.mark.parametrize(
        'value,   expected', [('rwx', 0o7), ('r-x', 0o5), ('r', 0o4), (None, 0o0), ('-', 0o0), ('wx', 0o3)]
    )
    def test_valid_permissions(self, value, expected):
        assert FilePermissionsBit(value).value == expected

    def test_toggle_permissions(self):
        b = FilePermissionsBit('r-x')
        b.can_read = False
        assert b.string_canonical == '--x'

        b.can_write = True
        assert b.string_canonical == '-wx'

        b.can_execute = False
        assert b.string_canonical == '-w-'


class TestFilePermissions:
    @pytest.mark.parametrize(
        'value,   expected',
        [
            ('rwxr-xr-x', 0o755),
            ('-rwxr--r--', 0o744),
            (('rwx', '', None), 0o700),
            ((6, 4, 4), 0o644),
            (('', 'rwx', 3), 0o073),
            (0o750, 0o750),
            ('640', 0o640),
            ('0o777', 0o777),
        ],
    )
    def test_valid_permissions(self, value, expected):
        if isinstance(value, tuple):
            assert FilePermissions(*value).decimal == expected
        else:
            assert FilePermissions(value).decimal == expected
