from xtl_core.errors import XTLCoreError


def test_root_is_an_exception():
    assert issubclass(XTLCoreError, Exception)
    assert not issubclass(XTLCoreError, (ValueError, TypeError, OSError))
