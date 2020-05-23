import pytest

import os
import shutil
import random

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
else:
    shutil.rmtree(CACHE_DIR)
    os.mkdir(CACHE_DIR)

@pytest.fixture(scope='session', params=[f'{random.randint(0, 9999):04d}' for _ in range(0, 3)])
def seed(request):
    """
    Creates 3 four-digit integer seeds and passes them to functions decorated with this fixture
    """
    return request.param