from xtl.exceptions import InvalidArgument, DimensionalityError
from xtl.math import si_units

from pint import UnitRegistry
from pint.errors import UndefinedUnitError

ureg = UnitRegistry()
"""
Main unit registry object
"""

ureg.define('weight_to_volume = 10 * gram / liter = %(w/v) = wv')

Q_ = ureg.Quantity


def smart_units(value, new_units, pretty=False):
    """
    Convert a value to :class:`pint.Quantity` object. Can parse numbers, strings and :class:`pint.Quantity` objects. It
    can be used to convert between compatible units, but also as a wrapper around values of ambiguous type.

    :param str or int or float or pint.Quantity value: Value to convert
    :param str or pint.Quantity new_units: Units to convert to
    :param bool pretty: Whether to reduce value to SI prefixes or not
    :return: Value in the new units
    :rtype: pint.Quantity
    :raise DimensionalityError: If the dimensions of `value` and `new_units` are different.
    :raise InvalidArgument: If the type of any argument is incorrect.
    """

    # Convert new_units to Quantity
    if isinstance(new_units, str):
        try:
            new_units = ureg.parse_expression(new_units)
        except UndefinedUnitError:
            raise InvalidArgument(raiser='new_units', message=f'{new_units}. Unknown units.')
    if not isinstance(new_units, Q_):
        raise InvalidArgument(raiser='new_units', message=f'{new_units}. Must be pint.Quantity or str.')

    # Convert value to Quantity
    if isinstance(value, str):
        value = Q_(value)
    elif isinstance(value, (int, float)):
        return Q_(value, new_units.units)
    if not isinstance(value, Q_):
        raise InvalidArgument(raiser='value', message=f'{value}. Must be pint.Quantity, str, int or float.')

    # Check dimensionality
    if value.dimensionality != new_units.dimensionality:
        raise DimensionalityError(src_units=value.units, dst_units=new_units.units)

    # Convert to new units
    value.ito(new_units)

    # Use appropriate prefix
    if pretty:
        prefix = si_units(value.magnitude)[-1]
        value.ito(f'{prefix}{value.units}')

    return value
