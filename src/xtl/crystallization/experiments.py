from dataclasses import dataclass, Field
import warnings

import numpy as np

from xtl.exceptions.warnings import ExistingReagentWarning


@dataclass
class _Reagent:
    name: str
    concentration: float
    unit: str = 'M'
    fmt_str: str = None
    solubility: float = None


@dataclass
class Reagent(_Reagent):
    ...


@dataclass
class ReagentWV(_Reagent):

    def __post_init__(self):
        self.unit = '%(w/v)'


@dataclass
class ReagentVV(_Reagent):

    def __post_init__(self):
        self.unit = '%(v/v)'


class CrystallizationExperiment:

    def __init__(self, shape: int | tuple[int, int]):
        self._data: np.array  # Concentrations array
        self._reagents = list()  # List of reagents
        self._shape: tuple[int, int]  # Shape of the crystallization experiment
        self._ndim: int  # Number of dimensions in shape

        # Shape of the crystallization experiment
        if isinstance(shape, int):
            self._shape = (shape, )
            self._ndim = 1
        elif isinstance(shape, (list, tuple)):
            if len(shape) != 2:
                raise ValueError(f'Invalid shape, must be of length 2, not {len(shape)}')
            self._shape = int(shape[0]), int(shape[1])
            self._ndim = 2
        else:
            raise TypeError('Invalid shape type, must be int or tuple')

        # Total number of conditions
        self.size = self._shape[0] * self._shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the crystallization experiment (rows, columns)
        :return:
        """
        return self._shape

    @property
    def no_rows(self) -> int:
        return self._shape[0]

    @property
    def no_columns(self) -> int:
        return self._shape[1]

    @property
    def no_wells(self):
        return self.size

    def add_reagent(self, reagent: Reagent | ReagentWV | ReagentVV):
        if not isinstance(reagent, _Reagent):
            raise TypeError('Invalid reagent type')
        if reagent not in self._reagents:
            self._reagents.append(reagent)
        else:
            warnings.warn(ExistingReagentWarning(raiser=reagent))

    def apply_reagent(self):
        ...
