import copy
from typing import Any, Iterable
from typing_extensions import Self

import pandas as pd
from pandas._typing import Dtype, ArrayLike, NpDtype

from xtl.datasets.scattering.data import ScatteringData
from xtl.scattering.metadata import *
from xtl.math.crystallography import radial_converters, unit_converters


class _ScatteringBase:

    def __init__(self,
                 # pd.DataFrame arguments
                 data: ArrayLike | Iterable | dict | pd.DataFrame | ScatteringData,
                 index: ArrayLike | pd.Index = None,
                 columns: ArrayLike | pd.Index = None,
                 types: Iterable | pd.Index | pd.Series | Dtype | str |
                        dict[str, Dtype | str] = None,
                 copy: bool = False,
                 # XTL arguments
                 wavelength: float = None,
                 metadata: ScatteringProfileMetadataType = None
                 ):
        self._data = ScatteringData(
            data=data, index=index, columns=columns, dtype=types, copy=copy, wavelength=wavelength
        )

        # Store metadata
        if metadata:
            if not isinstance(metadata, ScatteringProfileMetadata):
                raise TypeError(f'Expected {ScatteringProfileMetadata.__name__}, '
                                f'got {type(metadata)}')
            self._metadata = metadata
        else:
            self._metadata = None

    @property
    def metadata(self) -> ScatteringProfileMetadataType | None:
        """
        Metadata associated with the scattering profile when instantiated from a file.
        """
        return self._metadata

    def dropna(self, inplace: bool = False) -> Self:
        """
        Drop datapoints with missing data.
        """
        if inplace:
            self._data.dropna(inplace=True)
            return self
        else:
            r = copy.deepcopy(self)
            r._data.dropna(inplace=True)
            return r

    def drop_missing(self, inplace: bool = False) -> Self:
        """
        Drop datapoints with missing data.
        """
        return self.dropna(inplace=inplace)

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the scattering data array.
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the scattering data array.
        """
        return self._data.ndim

    @property
    def no_datapoints(self) -> int:
        """
        The number of datapoints in the array.
        """
        return self.shape[0]

    @property
    def no_columns(self) -> int:
        """
        The number of columns in the array.
        """
        return self.shape[1]

    @property
    def wavelength(self) -> float | None:
        """
        The wavelength of the scattering data array in Angstrom.
        """
        return self._data.wavelength

    @wavelength.setter
    def wavelength(self, wavelength: float | int) -> None:
        self._data.wavelength = wavelength

    @property
    def energy(self) -> float | None:
        """
        The energy of the scattering data array in keV.
        """
        return self._data.energy

    @energy.setter
    def energy(self, energy: float | int) -> None:
        self._data.energy = energy

class ScatteringProfile(_ScatteringBase):
    ...

class ScatteringProfiles(_ScatteringBase):
    ...