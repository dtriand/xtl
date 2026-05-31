from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from xtl.math.constants import KEV_PER_A
from xtl.math.crystallography import radial_converters, unit_converters
from xtl.units.base import Units, UnitsDescription


class RadialUnits(Units):
    TWOTHETA_DEG = (
        '2th_deg',
        UnitsDescription(
            name='2theta_degrees',
            desc='Scattering angle 2theta in degrees',
            repr='2theta (deg)',
            pretty='2\u03b8 (\u00b0)',
            latex=r'$2\theta\ (^\circ)$',
            aliases=('2th', '2theta', 'tth', 'ttheta', 'deg', 'degrees',
                     '2th_deg', '2th_degrees', '2theta_deg', '2theta_degrees',
                     'tth_deg', 'tth_degrees', 'ttheta_deg', 'ttheta_degrees'),
        )
    )

    TWOTHETA_RAD = (
        '2th_rad',
        UnitsDescription(
            name='2theta_radians',
            desc='Scattering angle 2theta in radians',
            repr='2theta (rad)',
            pretty='2\u03b8 (rad)',
            latex=r'$2\theta\ (\mathrm{rad})$',
            aliases=('rad', 'radians', '2th_rad', '2theta_rad', '2th_radians',
                     '2theta_radians', 'tth_rad', 'tth_radians', 'ttheta_rad',
                     'ttheta_radians')
        )
    )

    D_A = (
        'd_A',
        UnitsDescription(
            name='d_A',
            desc='d-spacing in Angstroms',
            repr='d (A)',
            pretty='d (\u212b)',
            latex=r'$d\ (\mathrm{\AA})$',
            aliases=('d', 'd_a', 'a', 'angstrom', 'angstroem',
                     'd_ang', 'd_angstrom', 'd_angstroem')
        )
    )

    D_NM = (
        'd_nm',
        UnitsDescription(
            name='d_nm',
            desc='d-spacing in nanometers',
            repr='d (nm)',
            pretty='d (nm)',
            latex=r'$d\ (\mathrm{nm})$',
            aliases=('d_nm', 'd_nanometers', 'nm', 'nanometers')
        )
    )

    Q_A = (
        'q_A^-1',
        UnitsDescription(
            name='q_1/A',
            desc='q-spacing (2 * pi / d) in reciprocal Angstroms',
            repr='q (1/A)',
            pretty='q (\u212b\u207B\u00B9)',
            latex=r'$q\ (\mathrm{\AA}^{-1})$',
            aliases=('q_a', 'q_1/a', '1/a', 'A^-1', 'q_ra', 'q_angstrom',
                     'q_angstroem', 'q_reciprocal_angstrom', 'q_reciprocal_angstroem')
        )
    )

    Q_NM = (
        'q_nm^-1',
        UnitsDescription(
            name='q_1/nm',
            desc='q-spacing (2 * pi / d) in reciprocal nanometers',
            repr='q (1/nm)',
            pretty='q (nm\u207B\u00B9)',
            latex=r'$q\ (\mathrm{nm}^{-1})$',
            aliases=('q', 'q_nm', 'q_nanometers', 'q_1/nm', '1/nm', 'nm^-1',
                     'q_nm^-1', 'q_rnm', 'q_reciprocal_nanometers')
        )
    )

    S_A = (
        's_A^-1',
        UnitsDescription(
            name='s_1/A',
            desc='s-spacing (1 / d) in reciprocal Angstroms',
            repr='s (1/A)',
            pretty='s (\u212b\u207B\u00B9)',
            latex=r'$s\ (\mathrm{\AA}^{-1})$',
            aliases=('s_a', 's_1/a', 's_ra', 's_angstrom', 's_angstroem',
                     's_reciprocal_angstrom', 's_reciprocal_angstroem')
        )
    )

    S_NM = (
        's_nm^-1',
        UnitsDescription(
            name='s_1/nm',
            desc='s-spacing (1 / d) in reciprocal nanometers',
            repr='s (1/nm)',
            pretty='s (nm\u207B\u00B9)',
            latex=r'$s\ (\mathrm{nm}^{-1})$',
            aliases=('s', 's_nm', 's_nanometers', 's_1/nm',
                     's_rnm', 's_reciprocal_nanometers')
        )
    )

    @property
    def quantity(self) -> str:
        return self.name.split('_')[0]

    @property
    def quantity_pretty(self) -> str:
        return self.pretty.split(' ')[0]

    @property
    def physical_units(self) -> str:
        return self.name.split('_')[1]

    @property
    def physical_units_pretty(self) -> str:
        return self.pretty.split(' ')[1].replace('(', '').replace(')', '')


@dataclass
class RadialValue:

    value: float | int | np.ndarray
    kind: RadialUnits

    def __post_init__(self):
        if isinstance(self.kind, str):
            # Recast kind to enum
            self.kind = RadialUnits(self.kind)

        # Get the quantity and (actual) units from the enum
        self._quantity, self._unit = self.kind.quantity, self.kind.physical_units

        # Standard radial units for conversions
        self._std_kinds = {
            '2theta': RadialUnits.TWOTHETA_DEG,
            'd': RadialUnits.D_A,
            'q': RadialUnits.Q_A,
            's': RadialUnits.S_A,
        }
        self._supported_quantities = set(self._std_kinds.keys())
        self._supported_units = {'degrees', 'radians', 'A', 'nm', '1/A', '1/nm'}

    def __repr__(self) -> str:
        return f'{self.value} {self.kind.repr}'

    def __rich_repr__(self):
        yield 'value', self.value
        yield 'units', self.kind.repr
        yield 'kind', self.kind

    def convert_to(self, units: RadialUnits | str, /, wavelength: Optional[float | int] = None,
           energy: Optional[float | int] = None) -> 'RadialValue':
        if isinstance(units, (str, RadialUnits)):
            target = RadialUnits(units)
        else:
            raise TypeError(f'`units` must be of type {RadialUnits.__name__} or str, got {type(units)}')

        # Validate target unit
        target_quantity, target_unit = target.quantity, target.physical_units
        if target_quantity not in self._supported_quantities:
            raise NotImplementedError(f'Unsupported radial quantity: {target_quantity!r}, '
                                      f'must be one of: {", ".join(self._supported_quantities)}')
        if target_unit not in self._supported_units:
            raise NotImplementedError(f'Unsupported unit: {target_unit!r}, '
                                      f'must be one of: {", ".join(self._supported_units)}')

        # Check if wavelength/energy is required for conversion
        _q0, _q1 = sorted([self._quantity, target_quantity])
        if _q0 == '2theta' and _q1 in {'d', 'q', 's'}:
            if wavelength is None and energy is None:
                raise ValueError(f'One of `wavelength` or `energy` must be specified to convert '
                                 f'from {self.kind.repr} to {target.repr}')
            elif wavelength is None:
                wavelength = KEV_PER_A / energy

        # Convert value
        f = self._conversion_function(kind0=self.kind, kind1=target)
        result = f(self.value, wavelength)
        return RadialValue(result, target)

    def _conversion_function(self, kind0: RadialUnits, kind1: RadialUnits) -> Callable:
        """
        Get the conversion function from kind0 to kind1.
        :param kind0: The original radial unit kind.
        :param kind1: The target radial unit kind.
        :return: A function that takes (value, wavelength) and returns the converted value.
        """
        quantity0, unit0 = kind0.quantity, kind0.physical_units
        quantity1, unit1 = kind1.quantity, kind1.physical_units

        # Same quantity, but different physical units
        if quantity0 == quantity1:
            if unit0 == unit1:
                return lambda x, w: x
            return lambda x, w: unit_converters[unit0][unit1](x)

        # Different quantities, need to convert to standard units first
        # Get standard units for each quantity
        std0 = self._std_kinds[quantity0].physical_units
        std1 = self._std_kinds[quantity1].physical_units

        # Convert original quantity to standard units
        if unit0 != std0:
            standardizer0 = lambda x, w: unit_converters[unit0][std0](x)
        else:
            standardizer0 = lambda x, w: x

        # Convert target quantity from standard units to requested units
        if unit1 != std1:
            standardizer1 = lambda x, w: unit_converters[std1][unit1](x)
        else:
            standardizer1 = lambda x, w: x

        # Get the converter between quantities (assuming standard units)
        converter = radial_converters[quantity0][quantity1]

        # Compose: standardizer0 (to std) -> converter (between quantities) -> standardizer1 (to target units)
        return lambda x, w: standardizer1(converter(standardizer0(x, w), w), w)
