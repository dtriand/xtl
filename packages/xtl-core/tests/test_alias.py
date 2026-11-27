import importlib
import pickle
import sys
import textwrap

import pytest

from xtl_core._alias import _AliasFinder


@pytest.fixture
def packages(tmp_path, monkeypatch):
    """
    A target package ``target_pkg`` and a package ``host_pkg`` whose ``alias`` subpackage
    aliases it. ``sys.meta_path`` and the new ``sys.modules`` entries are put back afterwards.
    """
    files = {
        'target_pkg/__init__.py': 'VALUE = 1\n',
        'target_pkg/sub.py': """
            class Thing:
                pass
        """,
        'target_pkg/pkg/__init__.py': '',
        'target_pkg/pkg/deep.py': 'DEEP = True\n',
        'host_pkg/__init__.py': '',
        'host_pkg/alias/__init__.py': """
            from xtl_core._alias import alias_package
            alias_package(__name__, 'target_pkg')
        """,
    }
    for name, content in files.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(textwrap.dedent(content))

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    modules = set(sys.modules)
    yield
    for name in set(sys.modules) - modules:
        del sys.modules[name]


@pytest.mark.usefixtures('packages')
class TestAliasPackage:
    def test_alias_is_target(self):
        import host_pkg.alias
        import target_pkg

        assert host_pkg.alias is target_pkg
        assert sys.modules['host_pkg.alias'] is target_pkg
        assert host_pkg.alias.VALUE == 1

    def test_submodule_alias_first(self):
        import host_pkg.alias.sub
        import target_pkg.sub

        assert host_pkg.alias.sub is target_pkg.sub
        assert sys.modules['host_pkg.alias.sub'] is target_pkg.sub

    def test_submodule_target_first(self):
        import target_pkg.sub
        import host_pkg.alias.sub

        assert host_pkg.alias.sub is target_pkg.sub

    def test_same_classes(self):
        from host_pkg.alias.sub import Thing as AliasThing
        from target_pkg.sub import Thing

        assert AliasThing is Thing
        assert Thing.__module__ == 'target_pkg.sub'

    def test_from_import(self):
        from host_pkg.alias import sub
        import target_pkg.sub

        assert sub is target_pkg.sub

    def test_nested_package(self):
        import host_pkg.alias.pkg.deep
        import target_pkg.pkg.deep

        assert host_pkg.alias.pkg is target_pkg.pkg
        assert host_pkg.alias.pkg.deep is target_pkg.pkg.deep

    def test_canonical_names(self):
        import host_pkg.alias.sub

        assert host_pkg.alias.sub.__name__ == 'target_pkg.sub'
        assert host_pkg.alias.sub.__spec__.name == 'target_pkg.sub'

    def test_lazy(self):
        import host_pkg.alias  # noqa: F401

        assert 'target_pkg.sub' not in sys.modules
        assert 'target_pkg.pkg' not in sys.modules

    def test_missing_module(self):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            import host_pkg.alias.missing  # noqa: F401
        assert exc_info.value.name == 'host_pkg.alias.missing'

    def test_reload(self):
        import host_pkg.alias.sub

        module = importlib.reload(host_pkg.alias.sub)
        assert module is sys.modules['target_pkg.sub']

    def test_pickle(self):
        from host_pkg.alias.sub import Thing

        assert type(pickle.loads(pickle.dumps(Thing()))) is Thing

    def test_one_finder(self):
        import host_pkg.alias  # noqa: F401
        from xtl_core._alias import alias_package

        alias_package('host_pkg.alias', 'target_pkg')
        finders = [f for f in sys.meta_path if isinstance(f, _AliasFinder) and f.alias == 'host_pkg.alias']
        assert len(finders) == 1
