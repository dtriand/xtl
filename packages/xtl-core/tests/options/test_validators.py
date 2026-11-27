import functools
import gc
from pathlib import Path
import tempfile

from hypothesis import given, strategies as st
from pydantic import AfterValidator, BeforeValidator
import pytest

from xtl_core.options import validators
from xtl_core.options.validators import (
    cast_as,
    cast_as_none_if_empty,
    cast_as_path_or_none,
    cast_as_temp_dir_if_none,
    validate_choices,
    validate_length,
    validate_path_exists,
    validate_path_is_absolute,
    validate_path_is_dir,
    validate_path_is_file,
)


@pytest.fixture
def tree(tmp_path) -> tuple[Path, Path, Path]:
    """An existing file, an existing directory and a missing path."""
    file = tmp_path / 'file.txt'
    file.touch()
    return file, tmp_path, tmp_path / 'missing'


class TestValidateLength:
    @given(value=st.lists(st.integers(), max_size=5), length=st.integers(0, 5))
    def test_lists(self, value, length):
        if len(value) == length:
            assert validate_length(value, length) is value
        else:
            with pytest.raises(ValueError, match=f'Expected length {length}, got {len(value)}'):
                validate_length(value, length)

    def test_not_iterable(self):
        assert validate_length(5, 3) == 5


class TestValidateChoices:
    def test_in_choices(self):
        assert validate_choices('a', ('a', 'b')) == 'a'

    def test_not_in_choices(self):
        with pytest.raises(ValueError, match=r"Value is not in choices: \('a', 'b'\)"):
            validate_choices('c', ('a', 'b'))


class TestValidatePathExists:
    def test_none(self):
        assert validate_path_exists(None) is None

    def test_exists(self, tree):
        file, directory, _ = tree
        assert validate_path_exists(file) == file
        assert validate_path_exists([file, directory]) == [file, directory]

    def test_missing(self, tree):
        file, _, missing = tree
        with pytest.raises(ValueError, match='Path does not exist'):
            validate_path_exists(missing)
        with pytest.raises(ValueError, match=r'Path\[1\] does not exist'):
            validate_path_exists([file, missing])

    def test_huge_path(self):
        with pytest.raises(ValueError, match=r'Path does not exist: .*\.\.\.$') as e:
            validate_path_exists(Path('x' * 10**5))
        assert len(str(e.value)) < 300


class TestValidatePathIsFile:
    def test_none(self):
        assert validate_path_is_file(None) is None

    def test_file(self, tree):
        file, _, _ = tree
        assert validate_path_is_file(file) == file
        assert validate_path_is_file([file]) == [file]

    @pytest.mark.parametrize('which', [1, 2], ids=['directory', 'missing'])
    def test_not_a_file(self, tree, which):
        with pytest.raises(ValueError, match='Path is not a file'):
            validate_path_is_file(tree[which])
        with pytest.raises(ValueError, match=r'Path\[1\] is not a file'):
            validate_path_is_file([tree[0], tree[which]])

    def test_huge_path(self):
        with pytest.raises(ValueError, match='Path is not a file'):
            validate_path_is_file(Path('x' * 10**5))


class TestValidatePathIsDir:
    def test_none(self):
        assert validate_path_is_dir(None) is None

    def test_directory(self, tree):
        _, directory, _ = tree
        assert validate_path_is_dir(directory) == directory
        assert validate_path_is_dir([directory]) == [directory]

    @pytest.mark.parametrize('which', [0, 2], ids=['file', 'missing'])
    def test_not_a_directory(self, tree, which):
        with pytest.raises(ValueError, match='Path is not a directory'):
            validate_path_is_dir(tree[which])
        with pytest.raises(ValueError, match=r'Path\[1\] is not a directory'):
            validate_path_is_dir([tree[1], tree[which]])

    def test_huge_path(self):
        with pytest.raises(ValueError, match='Path is not a directory'):
            validate_path_is_dir(Path('x' * 10**5))


class TestValidatePathIsAbsolute:
    def test_none(self):
        assert validate_path_is_absolute(None) is None

    def test_absolute(self, tree):
        file, directory, missing = tree
        for path in (file, directory, missing):
            assert validate_path_is_absolute(path) == path

    def test_relative(self):
        with pytest.raises(ValueError, match='Path is not absolute'):
            validate_path_is_absolute(Path('relative'))

    def test_list_of_absolute(self, tree):
        """Regression: in a list, paths were checked with ``is_file()`` instead of ``is_absolute()``."""
        _, directory, missing = tree
        assert validate_path_is_absolute([directory, missing]) == [directory, missing]

    def test_list_with_relative(self, tree, monkeypatch):
        """Regression: likewise, so a relative path to a file passed."""
        file, directory, _ = tree
        monkeypatch.chdir(directory)
        with pytest.raises(ValueError, match=r'Path\[0\] is not absolute'):
            validate_path_is_absolute([Path(file.name)])


class TestCastAs:
    @pytest.mark.parametrize('value, type_, expected', [('3', int, 3), (3, str, '3'), ('a', Path, Path('a'))])
    def test_cast(self, value, type_, expected):
        assert cast_as(value, type_) == expected

    def test_right_type_unchanged(self):
        value = Path('a')
        assert cast_as(value, Path) is value

    def test_callable(self):
        assert cast_as('3', lambda x: int(x) * 2) == 6

    def test_value_error(self):
        with pytest.raises(ValueError, match="Cannot cast 'abc' as int"):
            cast_as('abc', int)

    @pytest.mark.parametrize('value', [None, [1]], ids=['None', 'list'])
    def test_type_error(self, value):
        """Regression: a TypeError went through as it was."""
        with pytest.raises(ValueError, match='Cannot cast .* as int'):
            cast_as(value, int)

    def test_huge_input(self):
        with pytest.raises(ValueError) as e:
            cast_as('x' * 10**6, int)
        assert len(str(e.value)) < 300

    def test_callable_without_name(self):
        with pytest.raises(ValueError, match=r'Cannot cast None as functools\.partial'):
            cast_as(None, functools.partial(int, base=2))


class TestCastAsNoneIfEmpty:
    @pytest.mark.parametrize('value', ['', [], {}, 0, None])
    def test_empty(self, value):
        assert cast_as_none_if_empty(value) is None

    @pytest.mark.parametrize('value', ['a', [0], Path('.')])
    def test_not_empty(self, value):
        assert cast_as_none_if_empty(value) is value


class TestCastAsPathOrNone:
    @pytest.mark.parametrize('value', ['', None])
    def test_empty(self, value):
        assert cast_as_path_or_none(value) is None

    @pytest.mark.parametrize('value', ['a/b', Path('a', 'b')])
    def test_path(self, value):
        assert cast_as_path_or_none(value) == Path('a', 'b')


class TestCastAsTempDirIfNone:
    @pytest.fixture(autouse=True)
    def temp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
        return tmp_path

    def test_value(self, tmp_path):
        assert cast_as_temp_dir_if_none(str(tmp_path / 'given')) == tmp_path / 'given'
        assert not (tmp_path / 'given').exists()

    def test_prefix(self):
        assert cast_as_temp_dir_if_none(None, prefix='xtl_test_').name.startswith('xtl_test_')

    def test_none_in_the_temporary_directory(self, temp_dir):
        assert cast_as_temp_dir_if_none(None).parent == temp_dir

    def test_none_not_created(self, temp_dir):
        # Whoever uses the path creates it (a job, when it runs), so building a config leaves nothing behind
        path = cast_as_temp_dir_if_none(None)
        gc.collect()
        assert not path.exists()
        assert list(temp_dir.iterdir()) == []

    def test_new_each_time(self):
        assert cast_as_temp_dir_if_none(None) != cast_as_temp_dir_if_none(None)


_FACTORIES = [
    ('LengthValidator', (3,), AfterValidator, validate_length),
    ('ChoicesValidator', (('a',),), AfterValidator, validate_choices),
    ('PathExistsValidator', (), AfterValidator, validate_path_exists),
    ('PathIsFileValidator', (), AfterValidator, validate_path_is_file),
    ('PathIsDirValidator', (), AfterValidator, validate_path_is_dir),
    ('PathIsAbsoluteValidator', (), AfterValidator, validate_path_is_absolute),
    ('CastAsValidator', (int,), BeforeValidator, cast_as),
    ('CastAsNoneIfEmpty', (), BeforeValidator, cast_as_none_if_empty),
    ('CastAsPathOrNone', (), BeforeValidator, cast_as_path_or_none),
]


@pytest.mark.parametrize('factory, args, kind, func', _FACTORIES, ids=[f[0] for f in _FACTORIES])
def test_factories(factory, args, kind, func):
    validator = getattr(validators, factory)(*args)
    assert isinstance(validator, kind)
    assert getattr(validator.func, 'func', validator.func) is func
    assert factory in validators.__all__
