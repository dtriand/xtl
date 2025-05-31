import random
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Skip all tests in this directory."""
    for item in items:
        if item.path.parent.is_relative_to(item.session.fspath / 'tests' / 'GSAS2'):
            # Skip all tests in the GSAS2 directory
            item.add_marker(
                pytest.mark.skip(
                    reason='Deprecated: GSAS2 is no longer supported.'
                )
            )


# NOTE: Deprecated: GSAS2 is no longer supported.
# from xtl.GSAS2.GSAS2Interface import settings
from ..conftest import CACHE_DIR

iparam_synchrotron = """#GSAS-II instrument parameter file
Type:PXC
Bank:1.0
Lam:{}
Zero:{}
Polariz.:{}
Azimuth:{}
U:{}
V:{}
W:{}
X:{}
Y:{}
Z:{}
SH/L:{}"""

iparam_lab = """#GSAS-II instrument parameter file
Type:PXC
Bank:1.0
Lam1:{}
Lam2:{}
I(L2)/I(L1):{}
Zero:{}
Polariz.:{}
Azimuth:{}
U:{}
V:{}
W:{}
X:{}
Y:{}
Z:{}
SH/L:{}"""


class Iparam:

    def __init__(self, type, seed):
        self.type = type
        self.seed = seed

        random.seed(self.seed)
        rands = [random.random() for _ in range(0, 13)]
        if type == 'synchrotron':
            self.dictionary = iparam_synchrotron.format(*rands)
        elif type == 'laboratory':
            self.dictionary = iparam_lab.format(*rands)

    def save(self, f):
        with open(f, 'w') as fp:
            fp.write(self.dictionary)

    def save_with_error(self, f):
        lines = self.dictionary.split('\n')
        random.seed(self.seed)
        l = random.randrange(1, len(lines))  # cannot be the first line #GSAS-II
        with open(f, 'w') as fp:
            for i, line in enumerate(lines):
                if i != l:
                    fp.write(f'{line}\n')


@pytest.fixture(scope='package')
def gsas2_iparam_lab(seed):
    iparam_file = CACHE_DIR / f'lab_{seed}.instprm'
    if not iparam_file.exists():
        iparam = Iparam('laboratory', seed)
        iparam.save(iparam_file)
    return iparam_file


@pytest.fixture(scope='package')
def gsas2_iparam_synchrotron(seed):
    iparam_file = CACHE_DIR / f'syn_{seed}.instprm'
    if not iparam_file.exists():
        iparam = Iparam('synchrotron', seed)
        iparam.save(iparam_file)
    return iparam_file


@pytest.fixture(scope='package')
def gsas2_iparam_invalid(seed):
    iparam_file = CACHE_DIR / f'err_{seed}.instprm'
    if not iparam_file.exists():
        random.seed(seed)
        iparam = Iparam(random.choice(['laboratory', 'synchrotron']), seed)
        iparam.save_with_error(iparam_file)
    return iparam_file


@pytest.fixture(scope='package')
def setup_working_dir():
    settings.working_directory = CACHE_DIR
