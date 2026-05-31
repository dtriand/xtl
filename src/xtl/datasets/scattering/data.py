from collections.abc import Iterable, Mapping
from typing import Any, Optional

import pandas as pd
from pandas._typing import Dtype, ArrayLike

from xtl.datasets.scattering.dtypes import ScatteringAngleDtype, ANGULAR_DTYPES
from xtl.math.constants import KEV_PER_A
from xtl.units.scattering.radial import RadialUnits, RadialValue


class ScatteringData(pd.DataFrame):
    """
    A minimal pandas DataFrame subclass for scattering data.
    """

    # Propagate custom attributes through pandas operations
    _metadata = ['_radial_col', '_wavelength_A', '_energy_keV']


    def __init__(
            self,
            # pd.DataFrame arguments
            data: ArrayLike | Iterable | dict | pd.DataFrame | None = None,
            index: ArrayLike | pd.Index | None = None,
            columns: ArrayLike | Iterable | pd.Index | None = None,
            dtypes: Iterable | pd.Index | pd.Series | Dtype | str |
                   dict[str, Dtype | str] | None = None,
            copy: bool = False,
            # XTL arguments
            radial_col: Optional[str] = None,
            wavelength: Optional[float] = None,
            energy: Optional[float] = None
    ) -> None:
        if wavelength is not None and energy is not None:
            raise ValueError('Specify only one of wavelength and energy, not both')

        self._wavelength_A = None
        self._energy_keV = None

        super().__init__(data=data, index=index, columns=columns, dtype=None, copy=copy)

        # Set dtypes
        if dtypes is not None:
            if isinstance(dtypes, Mapping):
                dtype_items = dtypes.items()
            elif isinstance(dtypes, Iterable) and not isinstance(dtypes, (str, bytes)):
                dtypes = list(dtypes)
                if len(dtypes) != len(self.columns):
                    raise ValueError('Number of dtypes is not equal to number of columns')
                dtype_items = zip(self.columns, dtypes)
            else:
                dtype_items = ((col, dtypes) for col in self.columns)

            for col, dt in dtype_items:
                if col not in self.columns:
                    raise KeyError(f'Column {col!r} not found in data while setting dtypes')
                try:
                    self[col] = self[col].astype(dt, copy=False)
                except Exception as e:
                    raise TypeError(f'Error converting column {col!r} to dtype {dt!r}') from e

        self._radial_col = radial_col
        if self.has_radial and not isinstance(self.radial.dtype, ScatteringAngleDtype):
            raise TypeError(f'Radial column {self._radial_col} must have an angular dtype, '
                            f'got {self.radial.dtype}')

        # Copy over _metadata if present in provided data
        if isinstance(data, ScatteringData):
            self.__finalize__(data)

        if wavelength is not None:
            self.wavelength = wavelength
        elif energy is not None:
            self.energy = energy

    # Required pandas interface
    @property
    def _constructor(self) -> type['ScatteringData']:
        """Return the class constructor for subclass operations."""
        return ScatteringData

    @property
    def _constructor_sliced(self):
        return pd.Series

    # Custom properties
    @property
    def wavelength(self) -> Optional[float]:
        """Wavelength in Angstroms."""
        if self._wavelength_A:
            return self._wavelength_A
        elif self._energy_keV:
            return KEV_PER_A / self._energy_keV
        return None

    @wavelength.setter
    def wavelength(self, wavelength: Optional[float | int]) -> None:
        if wavelength is not None:
            self._wavelength_A = float(wavelength)
            self._energy_keV = KEV_PER_A / self._wavelength_A
        else:
            self._wavelength_A = None
            self._energy_keV = None

    @property
    def energy(self) -> Optional[float]:
        """Energy in keV."""
        if self._energy_keV:
            return self._energy_keV
        elif self._wavelength_A:
            return KEV_PER_A / self._wavelength_A
        return None

    @energy.setter
    def energy(self, energy: Optional[float | int]) -> None:
        if energy is not None:
            self._energy_keV = float(energy)
            self._wavelength_A = KEV_PER_A / self._energy_keV
        else:
            self._energy_keV = None
            self._wavelength_A = None

    @property
    def has_radial(self) -> bool:
        if self._radial_col is None:
            return False
        return self._radial_col in self.columns

    @property
    def radial(self) -> pd.Series:
        """The radial data column."""
        if self._radial_col not in self.columns:
            self._radial_col = None
        if self._radial_col is None:
            raise AttributeError('No radial column has been set')
        return self[self._radial_col]

    def convert_to(self, units: RadialUnits | str, inplace: bool = False) -> 'ScatteringData':
        if not self.has_radial:
            raise AttributeError('No radial column has been set')

        current = self.radial.dtype.units
        if current == units:
            return self if inplace else self.copy()

        # Transform to RadialValue for unit conversion
        radial = RadialValue(self.radial.values, current)
        converted = radial.convert_to(units, wavelength=self.wavelength)

        # Update with converted values
        result = self if inplace else self.copy()
        result[self._radial_col] = converted.value

        # Find and apply the correct dtype for the target unit type
        for dtype in ANGULAR_DTYPES.values():
            if dtype.units == units:
                target_dtype = dtype
                break
        else:
            raise NotImplementedError(f'No valid dtype implemented for {units}')
        result[self._radial_col] = result[self._radial_col].astype(target_dtype)

        return result
