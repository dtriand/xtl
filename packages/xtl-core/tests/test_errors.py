import pickle

import pytest
from pydantic import ValidationError

from xtl_core.errors import XTLCoreError
from xtl_core.options import Option, Options, OptionsValidationError


class Model(Options):
    count: int = Option(default=1)


def test_root_is_an_exception():
    assert issubclass(XTLCoreError, Exception)
    assert not issubclass(XTLCoreError, (ValueError, TypeError, OSError))


@pytest.mark.parametrize('base', [XTLCoreError, ValueError])
def test_options_validation_error(base):
    """Callers can catch it as the root, as a ValueError, or by name."""
    with pytest.raises(base) as excinfo:
        Model(count='abc')
    assert isinstance(excinfo.value, OptionsValidationError)
    assert not isinstance(excinfo.value, ValidationError)


def test_options_error_survives_pickling():
    # E.g. a job that fails in a worker process, whose result travels back pickled
    with pytest.raises(OptionsValidationError) as excinfo:
        Model(count='abc')
    restored = pickle.loads(pickle.dumps(excinfo.value))
    assert type(restored) is OptionsValidationError
    assert (restored.model_name, restored.errors) == (excinfo.value.model_name, excinfo.value.errors)
    assert str(restored) == str(excinfo.value)
