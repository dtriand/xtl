"""
Tests that need the network are skipped, not failed, when the network or the server is down (see `tests/conftest.py`).
"""

import types

import pytest
import requests

from tests.conftest import _skip_network_outages, network_outage


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f'{status} Error', response=response)


@pytest.mark.parametrize(
    'error',
    [requests.ConnectionError('down'), requests.Timeout('slow'), http_error(502), http_error(503), http_error(429)],
    ids=['connection', 'timeout', '502', '503', '429'],
)
def test_outages(error):
    assert network_outage(error)


@pytest.mark.parametrize(
    'error',
    [http_error(404), http_error(400), requests.HTTPError('no response'), AssertionError(), ValueError()],
    ids=['404', '400', 'no-response', 'assertion', 'other'],
)
def test_not_outages(error):
    assert network_outage(error) is None


def run(marked: bool, error: BaseException):
    """Raise `error` in a test, through the hook, as pytest would."""
    item = types.SimpleNamespace(get_closest_marker=lambda name: object() if marked else None)
    hook = _skip_network_outages(item)
    next(hook)
    hook.throw(error)


def test_outage_skips_a_marked_test():
    with pytest.raises(pytest.skip.Exception, match='Server unavailable: 502 Error'):
        run(marked=True, error=http_error(502))


def test_outage_fails_an_unmarked_test():
    with pytest.raises(requests.ConnectionError):
        run(marked=False, error=requests.ConnectionError('down'))


def test_other_errors_fail_a_marked_test():
    with pytest.raises(requests.HTTPError):
        run(marked=True, error=http_error(404))


def test_passing_test():
    item = types.SimpleNamespace(get_closest_marker=lambda name: object())
    hook = _skip_network_outages(item)
    next(hook)
    with pytest.raises(StopIteration) as stop:
        hook.send('result')
    assert stop.value.value == 'result'
