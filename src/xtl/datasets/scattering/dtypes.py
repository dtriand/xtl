"""
Minimal pandas extension dtypes for x-ray scattering data.

This module defines small, semantic pandas ExtensionDtype classes and a
lightweight ExtensionArray implementation used by the scattering dataset
to label columns such as intensities, uncertainties (sigmas) and angles.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype, take as ext_take

from xtl.units.scattering.radial import RadialUnits


class ScatteringDtype(ExtensionDtype):
    """
    Base ExtensionDtype for implementing persistent scattering data types
    """
    label: str
    label_pretty: str
    type = float
    kind = 'f'

    def __repr__(self) -> str:
        return f'{self.label}'

    @classmethod
    def construct_array_type(cls):
        return ScatteringArray


class IntensityDtype(ScatteringDtype):
    """Base dtype for scattering intensity data"""


class IntensitySigmaDtype(ScatteringDtype):
    """Base dtype for scattering intensity sigma data"""


class ScatteringAngleDtype(ScatteringDtype):
    """Base dtype for scattering angle data"""

    units: RadialUnits

    @property
    def label_pretty(self) -> str:
        return self.units.pretty


@register_extension_dtype
class IntensityArbitraryDtype(IntensityDtype):
    """Dtype for intensity in arbitrary units."""
    name = 'intensity_arbitrary'
    label = 'Intensity (A.U.)'
    label_pretty = 'I (A.U.)'


@register_extension_dtype
class IntensityAbsoluteDtype(IntensityDtype):
    """Dtype for intensity in absolute units."""
    name = 'intensity_absolute'
    label = 'Intensity (Abs)'
    label_pretty = 'I'

@register_extension_dtype
class IntensitySigmaArbitraryDtype(IntensitySigmaDtype):
    """Dtype for intensity uncertainties in arbitrary units."""
    name = 'intensity_sigma_arbitrary'
    label = 'Intensity Sigma (A.U.)'
    label_pretty = r'\sigma(I) (A.U.)'


@register_extension_dtype
class IntensitySigmaAbsoluteDtype(IntensitySigmaDtype):
    """Dtype for intensity uncertainties in absolute units."""
    name = 'intensity_sigma_absolute'
    label = 'Intensity Sigma (Abs)'
    label_pretty = r'\sigma(I)'


@register_extension_dtype
class Angle2ThetaDegDtype(ScatteringAngleDtype):
    """Dtype for scattering angle in 2theta (degrees)."""
    name = 'angle_2theta_deg'
    label = '2theta (deg)'
    units = RadialUnits.TWOTHETA_DEG


@register_extension_dtype
class Angle2ThetaRadDtype(ScatteringAngleDtype):
    """Dtype for scattering angle in 2theta (radians)."""
    name = 'angle_2theta_rad'
    label = '2theta (rad)'
    units = RadialUnits.TWOTHETA_RAD


@register_extension_dtype
class AngleDSpacingAngstromDtype(ScatteringAngleDtype):
    """Dtype for d-spacing (Angstroms)."""
    name = 'angle_d_spacing_A'
    label = 'd-spacing (A)'
    units = RadialUnits.D_A


@register_extension_dtype
class AngleDSpacingNanometerDtype(ScatteringAngleDtype):
    """Dtype for d-spacing (nanometers)."""
    name = 'angle_d_spacing_nm'
    label = 'd-spacing (nm)'
    units = RadialUnits.D_NM


@register_extension_dtype
class AngleSInverseAngstromDtype(ScatteringAngleDtype):
    """Dtype for scattering parameter s = 1/d (inverse Angstroms)."""
    name = 'angle_s_A'
    label = 's (1/A)'
    units = RadialUnits.S_A


@register_extension_dtype
class AngleSInverseNanometerDtype(ScatteringAngleDtype):
    """Dtype for scattering parameter s = 1/d (inverse nanometers)."""
    name = 'angle_s_nm'
    label = 's (1/nm)'
    units = RadialUnits.S_NM


@register_extension_dtype
class AngleQInverseAngstromDtype(ScatteringAngleDtype):
    """Dtype for scattering vector q = 2*pi/d (inverse Angstroms)."""
    name = 'angle_q_A'
    label = 'q (1/A)'
    units = RadialUnits.Q_A


@register_extension_dtype
class AngleQInverseNanometerDtype(ScatteringAngleDtype):
    """Dtype for scattering vector q = 2*pi/d (inverse nanometers)."""
    name = 'angle_q_nm'
    label = 'q (1/nm)'
    units = RadialUnits.Q_NM


ANGULAR_DTYPES: dict[str, ScatteringAngleDtype] = {
    dtype.units.value: dtype() for dtype in [
        Angle2ThetaDegDtype, Angle2ThetaRadDtype,
        AngleDSpacingAngstromDtype, AngleDSpacingNanometerDtype,
        AngleSInverseAngstromDtype, AngleSInverseNanometerDtype,
        AngleQInverseAngstromDtype, AngleQInverseNanometerDtype
    ]
}
"""Dictionary of scattering angle data types to their corresponding ``ScatteringDtype class``"""

INTENSITY_DTYPES: dict[str, IntensityDtype] = {
    dtype.name: dtype() for dtype in [
        IntensityArbitraryDtype, IntensityAbsoluteDtype
    ]
}
"""Dictionary of scattering intensity data types to their corresponding ``ScatteringDtype class``"""

SIGMA_DTYPES: dict[str, IntensitySigmaDtype] = {
    dtype.name: dtype() for dtype in [
        IntensitySigmaArbitraryDtype, IntensitySigmaAbsoluteDtype
    ]
}
"""Dictionary of scattering intensity sigma data types to their corresponding ``ScatteringDtype class``"""

SCATTERING_DTYPES: dict[str, ScatteringDtype] = ANGULAR_DTYPES | INTENSITY_DTYPES | SIGMA_DTYPES
"""Dictionary of scattering data type names to their corresponding ``ScatteringDtype class``"""


class ScatteringArray(ExtensionArray):
    """
    Lightweight ExtensionArray storing float64 values with a semantic dtype.

    Instances carry a reference to an ExtensionDtype subclass that provides a
    semantic name for the column (e.g. "intensity_arbitrary", "angle_q_nm").

    Operations between two ScatteringArray instances are allowed only when
    their associated ExtensionDtype classes are of the same type.
    """

    def __init__(self, values: np.ndarray, dtype: ScatteringDtype):
        if not isinstance(dtype, ScatteringDtype):
            raise TypeError(f'dtype must be a {ScatteringDtype.__name__}, got {dtype}')
        self._data = np.asarray(values, dtype=np.float64)
        self._dtype: ScatteringDtype = dtype

    # Required pandas interface
    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._data.nbytes

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        result = self._data[key]
        if np.ndim(result) == 0:  # scalar
            return float(result)
        elif isinstance(result, np.ndarray):
            return type(self)(result, self._dtype)
        return result

    def __setitem__(self, key, value):
        self._data[key] = value

    def isna(self):
        return np.isnan(self._data)

    def take(self, indices, *, allow_fill: bool = False, fill_value = None):
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        result = ext_take(
            self._data, indices,
            allow_fill=allow_fill,
            fill_value=fill_value if allow_fill else np.nan,
        )
        return type(self)(result, self._dtype)

    def copy(self):
        return type(self)(self._data.copy(), self._dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False):
        if dtype is None:
            raise TypeError(
                f'{ScatteringArray.__name__} requires an explicit {ScatteringDtype.__name__} dtype. '
                f'Pass dtype when constructing.'
            )
        if not isinstance(dtype, ScatteringDtype):
            raise TypeError(f'Expected {ScatteringDtype.__name__}, got {type(dtype)!r}')
        data = np.array(list(scalars), dtype=np.float64)
        return cls(data, dtype)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy: bool = False):
        scalars = [float(s) for s in strings]
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: ScatteringArray) -> ScatteringArray:
        return cls(values, original.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: list[ScatteringArray]) -> ScatteringArray:
        dtypes = {type(a.dtype) for a in to_concat}
        if len(dtypes) > 1:
            raise TypeError(
                f'Cannot concatenate {ScatteringArray.__name__}s with mixed dtypes: '
                f'{[a.dtype for a in to_concat]}'
            )
        return cls(np.concatenate([a._data for a in to_concat]), to_concat[0].dtype)

    # Helpers for numpy and pandas
    def __array__(self, dtype=None, copy=False):
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    def _values_for_argsort(self) -> np.ndarray:
        return self._data

    def _values_for_factorize(self):
        return self._data, np.nan

    def __repr__(self) -> str:
        return f'{ScatteringArray.__name__}(dtype={self._dtype.name!r}, n={len(self)})'

    # Arithmetic
    def _check_compat(self, other: Any) -> None:
        """Raise if *other* is also an ScatteringArray with a different dtype."""
        if isinstance(other, ScatteringArray) and type(self._dtype) is not type(other._dtype):
            raise TypeError(
                f'Cannot combine {self._dtype.name!r} and {other._dtype.name!r}: '
                'unit mismatch. Convert to a common unit first.'
            )

    def _arith(self, other: Any, op) -> ScatteringArray:
        self._check_compat(other)
        rhs = other._data if isinstance(other, ScatteringArray) else other
        return ScatteringArray(op(self._data, rhs), self._dtype)

    def _rarith(self, other: Any, op) -> ScatteringArray:
        self._check_compat(other)
        lhs = other._data if isinstance(other, ScatteringArray) else other
        return ScatteringArray(op(lhs, self._data), self._dtype)

    def __add__(self, other):
        return self._arith(other, np.add)

    def __radd__(self, other):
        return self._rarith(other, np.add)

    def __sub__(self, other):
        return self._arith(other, np.subtract)

    def __rsub__(self, other):
        return self._rarith(other, np.subtract)

    def __mul__(self, other):
        return self._arith(other, np.multiply)

    def __rmul__(self, other):
        return self._rarith(other, np.multiply)

    def __truediv__(self, other):
        return self._arith(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._rarith(other, np.true_divide)

    def __floordiv__(self, other):
        return self._arith(other, np.floor_divide)

    def __pow__(self, other):
        return self._arith(other, np.power)

    def __mod__(self, other):
        return self._arith(other, np.mod)

    def __neg__(self):
        return ScatteringArray(-self._data, self._dtype)

    def __abs__(self):
        return ScatteringArray(np.abs(self._data), self._dtype)

    def __eq__(self, other):
        if isinstance(other, ScatteringArray):
            self._check_compat(other)
            return self._data == other._data
        return self._data == other

    def __lt__(self, other):
        rhs = other._data if isinstance(other, ScatteringArray) else other
        return self._data < rhs

    def __le__(self, other):
        rhs = other._data if isinstance(other, ScatteringArray) else other
        return self._data <= rhs

    def __gt__(self, other):
        rhs = other._data if isinstance(other, ScatteringArray) else other
        return self._data > rhs

    def __ge__(self, other):
        rhs = other._data if isinstance(other, ScatteringArray) else other
        return self._data >= rhs
