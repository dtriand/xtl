import warnings

import numpy as np

from xtl.exceptions.warnings import ExistingReagentWarning
from .reagents import _Reagent, Reagent, ReagentWV, ReagentVV, Buffer
from .applicators import _ReagentApplicator, ConstantApplicator, GradientApplicator


class CrystallizationExperiment:

    def __init__(self, shape: int | tuple[int, int]):
        self._data: np.array = None  # Concentrations array (no_reagents, size)
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
    def no_wells(self) -> int:
        return self.size

    @property
    def data(self) -> np.array:
        return self._data.reshape((len(self._reagents), *self.shape))

    def add_reagent(self, reagent: Reagent | ReagentWV | ReagentVV | Buffer):
        if not isinstance(reagent, _Reagent):
            raise TypeError('Invalid reagent type')
        if reagent not in self._reagents:
            self._reagents.append(reagent)
        else:
            warnings.warn(ExistingReagentWarning(raiser=reagent))

    def apply_reagent(self, reagent: Reagent | ReagentWV | ReagentVV | Buffer,
                      applicator: ConstantApplicator | GradientApplicator,
                      location: str = 'everywhere'):
        if not isinstance(reagent, _Reagent):
            raise TypeError('Invalid reagent type')
        if not isinstance(applicator, _ReagentApplicator):
            raise TypeError('Invalid applicator type')

        # Determine shape from location
        shape = self._shape

        # Apply reagent to given shape
        reagent.applicator = applicator
        data = reagent.applicator.apply(shape)  # 1D array (size, )

        # Reformat the input data to fit the experiment shape
        pass

        # Append data to the concentrations array
        if self._data is None:
            self._data = data
        else:
            self._data = np.vstack((self._data, data))

        if reagent not in self._reagents:
            self._reagents.append(reagent)
        else:
            warnings.warn(ExistingReagentWarning(raiser=reagent))
