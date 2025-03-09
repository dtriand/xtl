from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from xtl.common.labels import Label
from xtl.math.crystallography import radial_converters, unit_converters


class RadialUnitType(Enum):
    TWOTHETA_DEG = '2th_deg'  # 2theta in degrees
    TWOTHETA_RAD = '2th_rad'  # 2theta in radians
    Q_NM = 'q_nm^-1'          # q in 1/nm
    Q_A = 'q_A^-1'            # q in 1/A
    D_NM = 'd_nm'             # d in nm
    D_A = 'd_A'               # d in A


@dataclass
class RadialUnit:
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
        return RadialUnitType(self.repr)

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
    def from_type(cls, r: RadialUnitType | str):
        if isinstance(r, str):
            r = RadialUnitType(r)
        if not isinstance(r, RadialUnitType):
            raise TypeError(f'Expected {RadialUnitType.__class__.__name__} or str, got {type(r)}')

        if r == RadialUnitType.TWOTHETA_DEG:
            return cls.ttheta_deg()
        elif r == RadialUnitType.TWOTHETA_RAD:
            return cls.ttheta_rad()
        elif r == RadialUnitType.Q_NM:
            return cls.q_nm()
        elif r == RadialUnitType.Q_A:
            return cls.q_A()
        elif r == RadialUnitType.D_NM:
            return cls.d_nm()
        elif r == RadialUnitType.D_A:
            return cls.d_A()
        else:
            raise ValueError(f'Unknown radial units: {r!r}')


@dataclass
class RadialValue:
    value: float | int
    type: RadialUnitType | str

    def __post_init__(self):
        r = RadialUnitType(self.type)
        self._radial: RadialUnit = RadialUnit.from_type(r)

        self._std_units = {
            '2theta': RadialUnit.ttheta_deg(),
            'd': RadialUnit.d_A(),
            'q':  RadialUnit.q_A()
        }
        self._supported_unit_types = list(self._std_units.keys())
        self._supported_units = ['deg', 'rad', 'A', 'nm', 'A^-1', 'nm^-1']

    @property
    def name(self):
        return self._radial.name

    @property
    def units(self):
        return self._radial.unit

    def to(self, units: RadialUnit | RadialUnitType | str, wavelength: Optional[float] = None) -> 'RadialValue':
        # Typecast to RadialUnit
        if isinstance(units, RadialUnitType) or isinstance(units, str):
            new = RadialUnit.from_type(units)
        else:
            new = units
        # Check if units is a RadialUnit
        if not isinstance(new, RadialUnit):
            raise TypeError(f'Expected {RadialUnit.__class__.__name__} or str, got {type(new)}')

        # Check if units are supported
        new: RadialUnit
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

    def _conversion_function(self, r0: RadialUnit, r1: RadialUnit) -> Callable:
        """
        Returns a number to multiply r0 to get r1.
        """
        u0, u1 = r0.unit.value, r1.unit.value
        t0, t1 = r0.name.value, r1.name.value

        # Check if units are the same
        if u0 == u1:
            return lambda x, w: x

        # Check if unit types are the same
        if t0 == t1:
            return lambda x, w: unit_converters[u0][u1](x)

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
