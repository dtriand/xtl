import pytest

from xtl_core.system.compatibility.py313 import path_exists, path_is_dir, path_is_file


@pytest.fixture
def tree(tmp_path):
    """An existing file, an existing directory and a missing path."""
    file = tmp_path / 'file.txt'
    file.touch()
    return file, tmp_path, tmp_path / 'missing'


# The `Path` methods of Python 3.14+: both must behave alike, but for paths that make them raise before 3.14
_CHECKS = [(path_exists, 'exists'), (path_is_file, 'is_file'), (path_is_dir, 'is_dir')]


@pytest.mark.parametrize('check, method', _CHECKS, ids=['exists', 'is-file', 'is-dir'])
class TestPathChecks:
    @pytest.mark.parametrize('which', [0, 1, 2], ids=['file', 'directory', 'missing'])
    def test_like_the_method(self, tree, check, method, which):
        path = tree[which]
        assert check(path) is getattr(path, method)()

    def test_str(self, tree, check, method):
        for path in tree:
            assert check(str(path)) is getattr(path, method)()

    @pytest.mark.parametrize('name', ['x' * 10**5, 'a\x00b'], ids=['huge', 'nul'])
    def test_any_name(self, check, method, name):
        # Too long a name raises `OSError` from the methods on Linux and macOS before Python 3.14
        assert check(name) is False
