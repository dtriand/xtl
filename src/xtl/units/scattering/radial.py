from dataclasses import dataclass
from typing import Callable, Optional

from xtl.common.labels import Label
from xtl.math.crystallography import radial_converters, unit_converters
from xtl.units.base import Units, UnitsDescription


class RadialUnits(Units):
    TWOTHETA_DEG = (
        '2th_deg',
        UnitsDescription(
            name='2th_deg',
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
            name='2th_rad',
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
            name='q_A^-1',
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
            name='q_nm^-1',
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
            name='s_A^-1',
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
            name='s_nm^-1',
            desc='s-spacing (1 / d) in reciprocal nanometers',
            repr='s (1/nm)',
            pretty='s (nm\u207B\u00B9)',
            latex=r'$s\ (\mathrm{nm}^{-1})$',
            aliases=('s', 's_nm', 's_nanometers', 's_1/nm',
                     's_rnm', 's_reciprocal_nanometers')
        )
    )


@dataclass
class RadialUnitDescription:
    name: Label
    unit: Label

    @property
    def repr(self):
        return f'{self.name.repr}_{self.unit.repr}'

    @property
    def latex(self):
        return f'{self.name.latex} ({self.unit.latex})'

    @property
    def type(self):
        return RadialUnits(self.repr)

    @classmethod
    def ttheta_deg(cls):
        return cls(name=Label(value='2theta', repr='2th', latex='2\u03b8'),
                   unit=Label(value='degrees', repr='deg', latex='\u00b0'))

    @classmethod
    def ttheta_rad(cls):
        return cls(name=Label(value='2theta', repr='2th', latex='2\u03b8'),
                   unit=Label(value='radians', repr='rad', latex='rad'))

    @classmethod
    def q_nm(cls):
        return cls(name=Label(value='q', repr='q', latex='q'),
                   unit=Label(value='1/nm', repr='nm^-1', latex='nm\u207B\u00B9'))

    @classmethod
    def q_A(cls):
        return cls(name=Label(value='q', repr='q', latex='q'),
                   unit=Label(value='1/A', repr='A^-1', latex='\u212b\u207B\u00B9'))

    @classmethod
    def d_nm(cls):
        return cls(name=Label(value='d', repr='d', latex='d'),
                   unit=Label(value='nm', repr='nm', latex='nm'))

    @classmethod
    def d_A(cls):
        return cls(name=Label(value='d', repr='d', latex='d'),
                   unit=Label(value='A', repr='A', latex='\u212b'))

    @classmethod
    def s_nm(cls):
        return cls(name=Label(value='s', repr='s', latex='s'),
                   unit=Label(value='1/nm', repr='nm^-1', latex='nm\u207B\u00B9'))

    @classmethod
    def s_A(cls):
        return cls(name=Label(value='s', repr='s', latex='s'),
                   unit=Label(value='1/A', repr='A^-1', latex='\u212b\u207B\u00B9'))

    @classmethod
    def from_type(cls, r: RadialUnits | str):
        if isinstance(r, str):
            r = RadialUnits(r)
        if not isinstance(r, RadialUnits):
            raise TypeError(f'Expected {RadialUnits.__class__.__name__} or str, got {type(r)}')

        if r == RadialUnits.TWOTHETA_DEG:
            return cls.ttheta_deg()
        elif r == RadialUnits.TWOTHETA_RAD:
            return cls.ttheta_rad()
        elif r == RadialUnits.Q_NM:
            return cls.q_nm()
        elif r == RadialUnits.Q_A:
            return cls.q_A()
        elif r == RadialUnits.D_NM:
            return cls.d_nm()
        elif r == RadialUnits.D_A:
            return cls.d_A()
        elif r == RadialUnits.S_NM:
            return cls.s_nm()
        elif r == RadialUnits.S_A:
            return cls.s_A()
        else:
            raise ValueError(f'Unknown radial units: {r!r}')


@dataclass
class RadialValue:
    value: float | int
    type: RadialUnits | str

    def __post_init__(self):
        if isinstance(self.type, str):
            # Recast type to enum
            self.type = RadialUnits(self.type)
        r = RadialUnits(self.type)
        self._radial: RadialUnitDescription = RadialUnitDescription.from_type(r)

        self._std_units = {
            '2theta': RadialUnitDescription.ttheta_deg(),
            'd': RadialUnitDescription.d_A(),
            'q':  RadialUnitDescription.q_A(),
            's': RadialUnitDescription.s_A()
        }
        self._supported_unit_types = list(self._std_units.keys())
        self._supported_units = ['deg', 'rad', 'A', 'nm', 'A^-1', 'nm^-1']

    @property
    def name(self):
        return self._radial.name

    @property
    def units(self):
        return self._radial.unit

    def to(self, units: RadialUnitDescription | RadialUnits | str, wavelength: Optional[float] = None) -> 'RadialValue':
        # Typecast to RadialUnit
        if isinstance(units, RadialUnits) or isinstance(units, str):
            new = RadialUnitDescription.from_type(units)
        else:
            new = units
        # Check if units is a RadialUnit
        if not isinstance(new, RadialUnitDescription):
            raise TypeError(f'Expected {RadialUnitDescription.__class__.__name__} or str, got {type(new)}')

        # Check if units are supported
        new: RadialUnitDescription
        if new.name.value not in self._supported_unit_types:
            raise ValueError(f'Unsupported radial units: {new.name.value!r}, choose one from: {",".join(self._supported_unit_types)}')
        if new.unit.repr not in self._supported_units:
            raise ValueError(f'Unsupported units: {new.unit.repr!r}, choose one from: {",".join(self._supported_units)}')

        # Check if wavelength is required for conversion
        if sorted([self.name.value, new.name.value]) in [['2theta', 'd'], ['2theta', 'q']]:
            if wavelength is None:
                raise ValueError(f'Wavelength is required to convert from {self.name.value} to {new.name.value}')

        f = self._conversion_function(self._radial, new)
        new_value = f(self.value, wavelength)
        return RadialValue(new_value, new.repr)

    def _conversion_function(self, r0: RadialUnitDescription, r1: RadialUnitDescription) -> Callable:
        """
        Returns a number to multiply r0 to get r1.
        """
        u0, u1 = r0.unit.value, r1.unit.value
        t0, t1 = r0.name.value, r1.name.value

        # Check if types are the same first (both 'q', both 'd', etc.)
        if t0 == t1:
            # Same type, just different units (if any)
            if u0 == u1:
                return lambda x, w: x
            return lambda x, w: unit_converters[u0][u1](x)

        # Different types (e.g., q vs s, or d vs 2theta)
        # Get factor f0 to convert r0 to standard units (2th, A, 1/A)
        f0 = self._conversion_function(r0, self._std_units[t0])

        # Get factor f1 to convert standard units (2th, A, 1/A) to r1 units
        f1 = self._conversion_function(self._std_units[t1], r1)

        # Get converter that assumes standard units
        converter = radial_converters[t0][t1]

        return lambda x, w: f1(converter(f0(x, w), w), w)

    def __repr__(self):
        return f'{self.value} {self.units.repr}'

    def __rich_repr__(self):
        yield 'value', self.value
        yield 'units', self.units.repr
        yield 'type', self._radial.type
