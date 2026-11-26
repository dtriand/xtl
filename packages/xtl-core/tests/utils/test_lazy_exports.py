import os.path

import pytest

from xtl_core.utils import strings
from xtl_core.utils.lazy_exports import lazy_dir, lazy_getattr, validate_exports

_EXPORTS = {
    'shorten': ('.strings', 'shorten'),  # relative to the package
    'join': ('os.path', 'join'),  # absolute
}


class TestLazyGetattr:
    @pytest.mark.parametrize('name, value', [('shorten', strings.shorten), ('join', os.path.join)])
    def test_resolves(self, name, value):
        namespace = {}
        assert lazy_getattr('xtl_core.utils', _EXPORTS, namespace, name) is value

    def test_caches(self):
        namespace = {}
        lazy_getattr('xtl_core.utils', _EXPORTS, namespace, 'shorten')
        assert namespace == {'shorten': strings.shorten}

    def test_cached(self):
        # A value already in the namespace is returned without importing
        marker = object()
        namespace = {'shorten': marker}
        assert lazy_getattr('xtl_core.utils', _EXPORTS, namespace, 'shorten') is marker

    def test_unknown(self):
        with pytest.raises(AttributeError, match="module 'pkg' has no attribute 'missing'"):
            lazy_getattr('pkg', _EXPORTS, {}, 'missing')

    def test_in_a_module(self):
        # As `__getattr__` and `__dir__` of a package
        import xtl_core.utils as package

        namespace = vars(package)
        assert 'join' not in namespace
        try:
            assert lazy_getattr(package.__name__, _EXPORTS, namespace, 'join') is os.path.join
            assert package.join is os.path.join
        finally:
            namespace.pop('join', None)


class TestLazyDir:
    def test_union_sorted(self):
        assert lazy_dir({'b': 1, '__name__': 'pkg'}, ['c', 'a', 'b']) == ['__name__', 'a', 'b', 'c']


class TestValidateExports:
    def test_valid(self):
        validate_exports('xtl_core.utils', _EXPORTS)

    def test_missing_attribute(self):
        with pytest.raises(AttributeError, match=r"Invalid lazy export 'x' -> \('os.path', 'nothing'\)"):
            validate_exports('xtl_core.utils', {'x': ('os.path', 'nothing')})

    def test_missing_module(self):
        with pytest.raises(ModuleNotFoundError):
            validate_exports('xtl_core.utils', {'x': ('.no_such_module', 'x')})
