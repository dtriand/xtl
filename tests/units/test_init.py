import pytest

from xtl.units import Q_, smart_units
from xtl.exceptions import DimensionalityError, InvalidArgument


class TestSmartUnits:

    def test_string_value_to_pint(self):
        assert smart_units('100 cm', 'm') == Q_(1, 'm')

    def test_number_value_to_pint(self):
        assert smart_units(1, 'm') == Q_(1, 'm')
        assert smart_units(1.5, 'm') == Q_(1.5, 'm')

    def test_pint_value_to_pint(self):
        assert smart_units(Q_(100, 'cm'), 'm') == Q_(1, 'm')

    def test_invalid_units(self):
        with pytest.raises(InvalidArgument, match='Must be pint.Quantity, str, int or float.'):
            smart_units([1, 'm'], 'cm')

    def test_invalid_dimensions(self):
        with pytest.raises(DimensionalityError):
            smart_units('1 m', 'L')