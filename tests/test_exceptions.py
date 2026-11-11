import warnings

import pytest

from xtl.exceptions import XTLError, XTLWarning
from xtl_core.errors import XTLCoreError


def test_xtl_error_is_a_core_error():
    """``except XTLCoreError`` catches the exceptions of both xtl and xtl-core (D-067)."""
    assert issubclass(XTLError, XTLCoreError)
    with pytest.raises(XTLCoreError):
        raise XTLError('raised by xtl')


def test_xtl_warning_is_a_user_warning():
    with pytest.warns(UserWarning, match='issued by xtl'):
        warnings.warn('issued by xtl', XTLWarning, stacklevel=1)
