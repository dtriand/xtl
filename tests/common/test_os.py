import pytest

from xtl.common.os import get_permissions_in_decimal


class TestPermissions:

    @pytest.mark.parametrize(
        'value,   expected', [
        ('0o755', 493),
        (777,     511),
        ('711',   457),
        pytest.param(755.,   None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param('asdf', None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(999,    None, marks=pytest.mark.xfail(raises=ValueError))
    ])
    def test_valid_permissions(self, value, expected):
        assert get_permissions_in_decimal(value) == expected