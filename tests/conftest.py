import random
from pathlib import Path
from shutil import rmtree

import pytest

CACHE_DIR = Path(__file__).parent / 'cache'

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()
else:
    rmtree(CACHE_DIR)
    CACHE_DIR.mkdir()


@pytest.fixture(scope='session', params=[f'{random.randint(0, 9999):04d}' for _ in range(0, 3)])
def seed(request):
    """
    Creates 3 four-digit integer seeds and passes them to functions decorated with this fixture
    """
    return request.param