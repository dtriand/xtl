import warnings

import numpy as np

from xtl.exceptions.warnings import ExistingReagentWarning
from .reagents import _Reagent, Reagent, ReagentWV, ReagentVV, Buffer
from .applicators import _ReagentApplicator, ConstantApplicator, GradientApplicator, StepFixedApplicator


class CrystallizationExperiment:

    def __init__(self, shape: int | tuple[int, int]):
        self._data: np.array = None  # Concentrations array (no_reagents, size)
        self._volumes: np.array = None  # Volumes array (no_reagents + 1, size)
        self._pH: np.array = None  # pH array (size, )
        self._reagents = list()  # List of reagents
        self._reagents_map: np.array = None  # Reagents map (no_reagents, size)
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

        # Initialize arrays
        self._pH = np.full(self.shape, np.nan)


    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the crystallization experiment (rows, columns)
        :return:
        """
        return self._shape

    @property
    def size(self) -> int:
        if self._ndim == 1:
            return self._shape[0]
        elif self._ndim == 2:
            return self._shape[0] * self._shape[1]

    @property
    def no_rows(self) -> int:
        return self._shape[0]

    @property
    def no_columns(self) -> int:
        return self._shape[1]

    @property
    def no_conditions(self) -> int:
        return self.size

    @property
    def data(self) -> np.array:
        return self._data.reshape((len(self._reagents), *self.shape))

    def _location_str_to_pos(self, location: str) -> list[int]:
        """
        Convert a string of locations to a list of positions.
        Examples: '1' -> [1],
                  '1-4' -> [1, 2, 3, 4],
                  '1,3,5' -> [1, 3, 5],
                  '1-3,6-7' -> [1, 2, 3, 6, 7]
        """
        groups = location.split(',')
        pos = []
        for group in groups:
            # Parse a range, e.g. group = '1-4'
            if '-' in group:
                start, stop = group.split('-')
                # Check if start and stop are numbers
                for value in (start, stop):
                    if not value.isdigit():
                        raise ValueError(f'Invalid location: \'{location}\'. '
                                         f'Element \'{value}\' is not a number.')
                # Add range to lines (including end value)
                pos.extend(range(int(start), int(stop) + 1))
            # Parse a single value, e.g. group = '1'
            else:
                # Check if value is a number
                if not group.isdigit():
                    raise ValueError(f'Invalid location: \'{location}\'. '
                                     f'Element \'{group}\' is not a number.')
                # Add range to lines
                pos.append(int(group))
        return pos

    def _location_to_indices(self, location: str | list) -> np.array:
        """
        Convert a string or list of locations to indices of the flattened data array (size, ). Examples of strings
        include: 'everywhere', 'all', 'row1', 'col1', 'row1-4', 'col1,3,5', 'row1,3-5', 'cell1', 'cell1-4', 'cell1,3,5'.
        The indices in the location string are interpreted as 1-based, while the returned indices are 0-based, in order
        to be congruent with self._data
        """
        if isinstance(location, str):
            # Parse strings such as 'everywhere', 'all', 'row1', 'col1', 'row1-4', 'col1,3,5', 'row1,3-5', 'cell1', etc.
            location = location.lower()  # Convert to lowercase
            if location in ('everywhere', 'all'):
                return np.arange(self.size)
            elif location.startswith('row') or location.startswith('col'):
                # e.g. row1, row1-4, col1,3,5, col1,3-5
                ltype = location[:3]
                lines = self._location_str_to_pos(location[3:])
                # Create a zero array with the shape of the experiment
                data = np.zeros(self.shape)
                lines = np.array(lines) - 1  # change to 0-based index
                if ltype == 'row':
                    indices_in_bounds = np.where(self.no_rows - lines > 0)[0]  # drop indices that are > no_rows
                    data[lines[indices_in_bounds], :] = 1.
                elif ltype == 'col':
                    indices_in_bounds = np.where(self.no_columns - lines > 0)[0]  # drop indices that are > no_columns
                    data[:, lines[indices_in_bounds]] = 1.
                # Get indices of flattened data
                indices = np.where(data.ravel() == 1.)[0]
                return indices
            elif location.startswith('cell'):
                cells = self._location_str_to_pos(location[4:])
                # Create a zero array with the shape of the experiment
                data = np.zeros(self.size)
                cells = np.array(cells) - 1  # change to 0-based index
                indices_in_bounds = np.where(self.no_conditions - cells > 0)[0]  # drop indices that are > no_conditions
                data[cells[indices_in_bounds]] = 1.
                # Get indices of flattened data
                indices = np.where(data == 1.)[0]
                return indices
        elif isinstance(location, (list, tuple)):
            # Parse lists such as ['cell1', 'row1-2'], [1, 4, 96], [(1, 1), (2, 3)]
            indices = np.full(self.size, False)
            for group in location:
                if isinstance(group, str):
                    # Strings are parsed by the previous block
                    indices[self._location_to_indices(group)] = True
                elif isinstance(group, int):
                    # Integers are interpreted as 1-based cell indices
                    if group > self.size or group < 1:  # ignore out of bounds indices
                        continue
                    indices[group - 1] = True
                elif isinstance(group, (list, tuple)):
                    # Lists are interpreted as 1-based (row, column) cell coordinates
                    if len(group) != 2:  # ignore any list with more or less than 2 elements
                        continue
                    if not (isinstance(group[0], int) and isinstance(group[1], int)):  # ignore non-numbers
                        continue
                    row, col = group  # 1-based row and column coordinates
                    if row > self.no_rows or col > self.no_columns:  # ignore out of bounds indices
                        continue
                    i = (row - 1) * self.no_columns + col  # cell index, 1-based
                    indices[i - 1] = True
            return np.where(indices == True)[0]

    def apply_reagent(self, reagent: Reagent | ReagentWV | ReagentVV | Buffer,
                      applicator: ConstantApplicator | GradientApplicator | StepFixedApplicator,
                      pH_applicator: ConstantApplicator | GradientApplicator | StepFixedApplicator = None,
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

    def calculate_volumes(self, final_volume: float | int):
        # V1 = C2 * V2 / C1
        c_stocks = np.array([reagent.concentration for reagent in self._reagents])
        v_stocks = ((self._data * final_volume).T / c_stocks).T
        v_water = final_volume - np.sum(v_stocks, axis=0)

        impossibles = np.where(v_water < 0)[0]
        if impossibles.size > 0:
            raise ValueError(f'Impossible condition: Negative volume of water in well(s) {impossibles}')

        self._volumes = np.vstack((v_stocks, v_water))
        return self._volumes
