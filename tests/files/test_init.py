import xtl.files
from xtl.common.lazy_exports import validate_exports


def test_exports():
    validate_exports('xtl.files', xtl.files._EXPORTS)
    assert sorted(xtl.files._EXPORTS) == sorted(xtl.files.__all__)


def test_jobs_is_not_exported():
    # The xtl.files.jobs subpackage is imported on its own, not as a lazy export
    assert 'jobs' not in xtl.files.__all__
    assert 'jobs' not in xtl.files._EXPORTS
