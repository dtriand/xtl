from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
import toml
from toml.decoder import CommentValue

from xtl_core.options.toml import ExtendedTomlEncoder
from xtl_core.system.os.permissions import FilePermissions


def dumps(data: dict) -> str:
    return toml.dumps(data, encoder=ExtendedTomlEncoder())


class TestExtendedTomlEncoder:
    @pytest.mark.parametrize(
        'path', [Path('a', 'b'), PurePosixPath('a/b'), PureWindowsPath('a\\b')], ids=['Path', 'posix', 'windows']
    )
    def test_paths_posix(self, path):
        assert dumps({'path': path}) == 'path = "a/b"\n'

    @pytest.mark.xfail(strict=True, reason='Paths are written between quotes without escaping')
    @pytest.mark.parametrize('name', ['a"b', 'a\\b'], ids=['quote', 'backslash'])
    def test_paths_escaped(self, name):
        assert toml.loads(dumps({'path': PurePosixPath(name)})) == {'path': name}

    def test_file_permissions(self):
        assert dumps({'mode': FilePermissions(0o755)}) == 'mode = "755"\n'

    def test_none(self):
        assert ExtendedTomlEncoder().dump_value(None) == '""'

    def test_comments(self):
        value = CommentValue(val=1, comment='# one', beginline=False, _dict=dict)
        assert dumps({'a': value}) == 'a = 1 # one\n'
