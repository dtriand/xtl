from xtl.GSAS2.parameters import InstrumentalParameters
from xtl.exceptions import InvalidArgument, FileError
# from ..conftest import CACHE_DIR

import os
import random
import pytest
from pyxray import NotFound

class TestInstrumentalParameters:

    class TestLoadFile:
        def test_load_success(self, gsas2_iparam_lab, gsas2_iparam_synchrotron):
            # Randomized files created by conftest.py
            ipl = InstrumentalParameters(file=gsas2_iparam_lab)
            ips = InstrumentalParameters(file=gsas2_iparam_synchrotron)
            assert ipl._initial_dict
            assert ips._initial_dict

        def test_load_fail(self, gsas2_iparam_invalid):
            with pytest.raises(FileError, match='Could not find'):
                # Randomized files created by conftest.py
                assert InstrumentalParameters(file=gsas2_iparam_invalid)

    class TestDefaultsSynchrotron:
        def test_defaults(self):
            ips = InstrumentalParameters.defaults_synchrotron()
            assert ips.wavelength == 1

        def test_wavelength(self):
            ips = InstrumentalParameters.defaults_synchrotron(1.3)
            assert ips.wavelength == 1.3

        def test_fail(self):
            with pytest.raises(InvalidArgument, match='Must be a positive number'):
                InstrumentalParameters.defaults_synchrotron('1')
            with pytest.raises(InvalidArgument, match='Must be a positive number'):
                InstrumentalParameters.defaults_synchrotron(-0.3)

    class TestDefaultsLab:
        def test_defaults(self):
            ipl = InstrumentalParameters.defaults_lab()
            assert ipl.source == 'CuKa(tabulated)'

        def test_element(self):
            for element in [12, 18, 22, 45, 'ni', 'li', 'Mg', 'Ti']:
                ipl = InstrumentalParameters.defaults_lab(element)
                assert ipl._initial_dict

        def test_fail(self):
            with pytest.raises(InvalidArgument, match='Unknown element'):
                InstrumentalParameters.defaults_lab('uk')

            with pytest.raises(InvalidArgument, match='Unknown element'):
                InstrumentalParameters.defaults_lab('coper')

            with pytest.raises(InvalidArgument, match='Unknown atomic number'):
                InstrumentalParameters.defaults_lab(200)

            with pytest.raises(InvalidArgument, match='Characteristic radiation could not be computed for element'):
                InstrumentalParameters.defaults_lab(100)

    class TestSetters:
        def test_type(self, gsas2_iparam_lab, gsas2_iparam_synchrotron):
            ipl = InstrumentalParameters(file=gsas2_iparam_lab)
            ips = InstrumentalParameters(file=gsas2_iparam_synchrotron)
            new_type = ''.join([random.choice('asdfghjkl') for _ in range(0, 5)])
            ipl.type = new_type
            ips.type = new_type
            assert ipl.type == new_type
            assert ips.type == new_type

            with pytest.raises(InvalidArgument, match='str'):
                ipl.type = 1

            with pytest.raises(InvalidArgument, match='str'):
                ips.type = 1

        def test_bank(self, gsas2_iparam_lab, gsas2_iparam_synchrotron):
            ipl = InstrumentalParameters(file=gsas2_iparam_lab)
            ips = InstrumentalParameters(file=gsas2_iparam_synchrotron)
            new_bank = float(random.randint(1, 10))
            ipl.bank = new_bank
            ips.bank = new_bank
            assert ipl.bank == new_bank
            assert ips.bank == new_bank

            with pytest.raises(InvalidArgument, match='number'):
                ipl.bank = '1'

            with pytest.raises(InvalidArgument, match='number'):
                ips.bank = '1'

        def test_wavelength_syn(self, gsas2_iparam_synchrotron):
            ips = InstrumentalParameters(file=gsas2_iparam_synchrotron)

            new_wavelength = random.randint(1, 500) / 100  # between 0.01 and 5
            ips.wavelength = new_wavelength
            assert ips.wavelength == new_wavelength

            with pytest.raises(InvalidArgument, match='positive number'):
                ips.wavelength = -1.2

            with pytest.raises(InvalidArgument, match='positive number'):
                ips.wavelength = 0

        def test_wavelength_lab(self, gsas2_iparam_lab):
            ipl = InstrumentalParameters(file=gsas2_iparam_lab)

            new_wavelength = [random.randint(1, 500) / 100 for _ in range(0, 2)] + [random.random()]
            ipl.wavelength = new_wavelength
            assert ipl.wavelength == tuple(new_wavelength)

            with pytest.raises(InvalidArgument, match='iterable of three positive numbers'):
                new_wavelength[random.randint(0, 2)] *= -1  # negative value
                ipl.wavelength = new_wavelength

            with pytest.raises(InvalidArgument, match='iterable of three positive numbers'):
                new_wavelength = random.random()  # not iterable of length 3
                ipl.wavelength = new_wavelength

        def test_zero_shift(self, gsas2_iparam_lab, gsas2_iparam_synchrotron):
            ipl = InstrumentalParameters(file=gsas2_iparam_lab)
            ips = InstrumentalParameters(file=gsas2_iparam_synchrotron)
            new_zero = float(random.random())
            ipl.zero_shift = new_zero
            ips.zero_shift = new_zero
            assert ipl.zero_shift == new_zero
            assert ips.zero_shift == new_zero

            with pytest.raises(InvalidArgument, match='float or int'):
                ipl.zero_shift = '1'

            with pytest.raises(InvalidArgument, match='float or int'):
                ips.zero_shift = '1'

        def test_polarization(self, gsas2_iparam_lab, gsas2_iparam_synchrotron):
            ipl = InstrumentalParameters(file=gsas2_iparam_lab)
            ips = InstrumentalParameters(file=gsas2_iparam_synchrotron)
            new_polar = float(random.randint(50, 100) / 100)  # between 0.5 and 1.0
            ipl.polarization = new_polar
            ips.polarization = new_polar
            assert ipl.polarization == new_polar
            assert ips.polarization == new_polar

            with pytest.raises(InvalidArgument, match='number between 0.5 and 1.0'):
                ipl.polarization = '1'

            with pytest.raises(InvalidArgument, match='number between 0.5 and 1.0'):
                ips.polarization = 0.3

