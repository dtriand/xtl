import warnings

from xtl.cli.utils import Catcher


def test_catches_and_reports():
    printed = []
    with Catcher(echo_func=printed.append, error_message='Failed:') as catcher:
        raise ValueError('boom')
    assert catcher.raised
    (error,) = catcher.errors
    assert error[0] is ValueError
    assert str(error[1]) == 'boom'
    assert printed[0] == 'Failed:'
    assert any('ValueError: boom' in line for line in printed)


def test_silent():
    printed = []
    with Catcher(echo_func=printed.append, silent=True) as catcher:
        raise ValueError('boom')
    assert catcher.raised
    assert printed == []


def test_traceback_func():
    tracebacks = []
    with Catcher(echo_func=print, traceback_func=tracebacks.append):
        raise KeyError('missing')
    (exc,) = tracebacks
    assert isinstance(exc, KeyError)


def test_no_exception():
    with Catcher(echo_func=print) as catcher:
        pass
    assert not catcher.raised
    assert catcher.errors == []


def test_warnings_recorded_and_reported():
    printed = []
    with Catcher(echo_func=printed.append) as catcher:
        warnings.warn('careful', UserWarning, stacklevel=1)
    assert not catcher.raised
    assert [str(w.message) for w in catcher.warnings] == ['careful']
    assert '    UserWarning: careful' in printed
