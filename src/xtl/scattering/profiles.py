from copy import deepcopy

from typing import TYPE_CHECKING
import numpy as np
from typing_extensions import Self

import pandas as pd
from pandas._typing import Dtype, ArrayLike

from xtl.units.base import Units

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from xtl.datasets.scattering.data import ScatteringData
from xtl.datasets.scattering.dtypes import ScatteringAngleDtype, IntensityDtype, IntensitySigmaDtype
from xtl.scattering.metadata import *
from xtl.units.scattering.radial import RadialUnits, RadialUnitsDescription


class _ScatteringBase:

    def __init__(self,
                 # pd.DataFrame arguments
                 data: ArrayLike | Iterable | dict | pd.DataFrame | ScatteringData,
                 /,
                 index: ArrayLike | pd.Index | None = None,
                 columns: ArrayLike | pd.Index | Iterable | None = None,
                 dtypes: Iterable | pd.Index | pd.Series | Dtype | str |
                        dict[str, Dtype | str] | None = None,
                 copy: bool = False,
                 # XTL arguments
                 radial_col: str | None = None,
                 wavelength: float | None = None,
                 energy: float | None = None,
                 metadata: ScatteringProfileMetadataType | None = None
                 ):
        self._data = ScatteringData(
            data=data, index=index, columns=columns, dtypes=dtypes, copy=copy, wavelength=wavelength, energy=energy,
            radial_col=radial_col
        )

        # Check for at least one radial dtype
        if not any([isinstance(dt, ScatteringAngleDtype) for dt in self._data.dtypes]):
            raise TypeError('At least one column must have an angular dtype')

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
            r = deepcopy(self)
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

    def __init__(self,
                 data: ArrayLike | Iterable | dict | pd.DataFrame | ScatteringData,
                 /,
                 index: ArrayLike | pd.Index | None = None,
                 columns: ArrayLike | pd.Index | Iterable | None = None,
                 dtypes: Iterable[Dtype | str] | None = None,
                 copy: bool = False,
                 radial_col: str | None = None,
                 wavelength: float | None = None,
                 energy: float | None = None,
                 metadata: ScatteringProfileMetadataType | None = None
                 ):
        super().__init__(
            data, index=index, columns=columns, dtypes=dtypes, copy=copy, radial_col=radial_col, wavelength=wavelength,
            energy=energy, metadata=metadata
        )

        # Check shape
        ncols = self._data.shape[1]
        if ncols < 2 or ncols > 3:
            raise ValueError(f'{self.__class__.__name__} must have 2 or 3 columns: radial, intensity, and optional '
                             f'sigma, not {self._data.shape[1]}')

        # Identify column types
        self._cols: dict[str, str | None] = {'x': None, 'y': None, 'e': None}
        for col in self._data.columns:
            dtype = self._data[col].dtype
            if isinstance(dtype, ScatteringAngleDtype):
                self._cols['x'] = col
            elif isinstance(dtype, IntensityDtype):
                self._cols['y'] = col
            elif isinstance(dtype, IntensitySigmaDtype):
                self._cols['e'] = col

        # Check for required columns
        if self._cols['x'] is None:
            raise ValueError(f'`data` must contain a radial column')
        if self._cols['y'] is None:
            raise ValueError(f'`data` must contain an intensity column')
        if self._cols['e'] is None and ncols == 3:
            raise ValueError(f'`data` must contain a sigma column')

    @property
    def data(self) -> ScatteringData:
        return self._data

    @property
    def radial(self) -> pd.Series:
        return self._data[self._cols['x']]

    @property
    def x(self) -> pd.Series:
        return self.radial

    @property
    def intensity(self) -> pd.Series:
        return self._data[self._cols['y']]

    @property
    def sigma(self) -> pd.Series | None:
        if self._cols['e'] is not None:
            return self._data[self._cols['e']]
        else:
            return None

    def get_radial_array(self, units: RadialUnits | str = RadialUnits.Q_NM) -> np.ndarray:
        if units == self._data.radial.dtype.units:
            return self.radial.values.to_numpy(np.float64)
        return self._data.convert_to(units, inplace=False)[self._cols['x']].values.to_numpy(np.float64)

    def get_guinier_array(self, units: RadialUnits | str = RadialUnits.Q_NM) -> np.ndarray:
        q = self.get_radial_array(units=units)
        x = q**2

        I_positive = self.intensity > 0
        y = np.full_like(self.intensity, np.nan)
        y[I_positive] = np.log(self.intensity[I_positive])

        e = np.full_like(self.intensity, np.nan)
        if self.sigma is not None:
            e[I_positive] = self.sigma[I_positive].astype(np.float64) / self.intensity[I_positive].astype(np.float64)

        return np.array([x, y, e])

    def get_kratky_array(self, units: RadialUnits | str = RadialUnits.Q_NM) -> np.ndarray:
        q = self.get_radial_array(units=units)
        x = q**2

        y = self.intensity.astype(np.float64) * x

        if self.sigma is not None:
            e = self.sigma.astype(np.float64) * x
        else:
            e = np.full_like(self.intensity, np.nan)

        return np.array([x, y, e])
    
    def get_porod_array(self, units: RadialUnits | str = RadialUnits.Q_NM) -> np.ndarray:
        q = self.get_radial_array(units=units)
        x = q**4

        y = self.intensity.astype(np.float64) * x

        if self.sigma is not None:
            e = self.sigma.astype(np.float64) * x
        else:
            e = np.full_like(self.intensity, np.nan)

        return np.array([x, y, e])

    def _format_plot_axes(self, ax: 'plt.Axes', units_x: RadialUnits | None,
                          units_y: Units | None = None, title: str | None = None, **kwargs) -> None:
        # Set ticks
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)

        # Set labels
        if units_x:
            ax.set_xlabel(units_x.latex)
        if units_y:
            ax.set_ylabel(units_y.latex)
        if title:
            ax.set_title(title)

    def plot(self, ax: 'plt.Axes' = None, units: RadialUnits | str = RadialUnits.Q_NM, errors: bool = False,
             **kwargs) -> 'plt.Axes':
        import matplotlib.pyplot as plt

        units = RadialUnits(units)
        kwargs.setdefault('title', 'SAXS profile')
        label = kwargs.get('label', self.metadata.name if self.metadata is not None else None)
        color = kwargs.get('color', None)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        x = self.get_radial_array(units=units)
        line = ax.plot(x, self.intensity, label=label, color=color)
        if errors and self.sigma is not None:
            ax.fill_between(x, self.intensity - self.sigma, self.intensity + self.sigma,
                            label=label, color=line[0].get_color(), alpha=0.3, edgecolor=None,
                            zorder=line[0].get_zorder() - 1)

        self._format_plot_axes(ax, units_x=units, units_y=self.intensity.dtype.units)

        return ax