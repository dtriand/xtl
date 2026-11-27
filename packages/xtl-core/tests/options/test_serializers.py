import pytest

from xtl_core.options.serializers import PermissionOctal
from xtl_core.system.os.permissions import FilePermissions


@pytest.mark.parametrize('mode, expected', [(0o755, '755'), (0o640, '640'), (0o007, '007')])
def test_permission_octal(mode, expected):
    assert PermissionOctal(FilePermissions(mode)) == expected
