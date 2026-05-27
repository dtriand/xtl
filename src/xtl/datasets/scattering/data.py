from typing import Any, Optional, Iterable

import pandas as pd
from pandas._typing import Dtype, ArrayLike

from xtl.datasets.scattering.dtypes import ScatteringAngleDtype
from xtl.math.constants import KEV_PER_A


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
            columns: ArrayLike | pd.Index | None = None,
            dtype: Iterable | pd.Index | pd.Series | Dtype | str |
                   dict[str, Dtype | str] | None = None,
            copy: bool = False,
            # XTL arguments
            radial_col: Optional[str] = None,
            wavelength: Optional[float] = None,
            energy: Optional[float] = None
    ) -> None:
        if wavelength is not None and energy is not None:
            raise ValueError('Specify only one of wavelength and energy, not both')

        self._radial_col = radial_col
        self._wavelength_A = None
        self._energy_keV = None

        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )

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
    def radial(self) -> pd.Series:
        """The radial data column."""
        if self._radial_col not in self.columns:
            self._radial_col = None
        if self._radial_col is None:
            raise AttributeError('No radial column has been set')
        return self[self._radial_col]

    @property
    def radial_dtype(self) -> ScatteringAngleDtype | None:
        if self._radial_col is None:
            return None
        d = self.radial.dtype
        return d if isinstance(d, ScatteringAngleDtype) else None

    # radial conversions
    # TODO: Rework xtl.units.crystallography.radial.RadialValue
