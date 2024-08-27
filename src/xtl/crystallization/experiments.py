import warnings
from types import NoneType

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
        self._pH = np.full(self.size, np.nan)


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

    def _index_1D_to_2D(self, i: int | np.ndarray[int]) -> tuple[int, int] | np.ndarray[int, int]:
        """
        Convert a 1D index to 2D indices (both 0-based)
        """
        if isinstance(i, int):
            # Check if index is within bounds
            if i > self.size:
                raise ValueError(f'Index {i} is out of bounds')
            # Calculate row and column
            col = i % self.no_columns
            row = int((i - col) / self.no_columns)
            # Double check if the results are within bounds
            if row > self.no_rows:
                raise ValueError(f'Index {i} is out of bounds: row index {row} is larger than the number of rows')
            if col > self.no_columns:
                raise ValueError(f'Index {i} is out of bounds: column index {col} is larger than the number of columns')
            return row, col
        elif isinstance(i, np.ndarray):
            # Check if indices are within bounds
            out_of_range = np.where(i > self.size)[0]
            if out_of_range.size > 0:
                raise ValueError(f'Indices {i[out_of_range]} are out of bounds')
            # Calculate row and column
            col = np.mod(i, self.no_columns)
            row = np.divide(i - col, self.no_columns).astype(int)
            # Double check if the results are within bounds
            rows_out_of_range = np.where(row > self.no_rows)[0]
            if rows_out_of_range.size > 0:
                raise ValueError(f'Indices {i[rows_out_of_range]} are out of bounds: '
                                 f'row index is larger than the number of rows')
            cols_out_of_range = np.where(col > self.no_columns)[0]
            if cols_out_of_range.size > 0:
                raise ValueError(f'Indices {i[cols_out_of_range]} are out of bounds: '
                                 f'column index is larger than the number of columns')
            return np.vstack((row, col)).T
        else:
            raise TypeError(f'Incompatible type for \'i\': {type(i)}')

    def _index_2D_to_1D(self, row: int | np.ndarray[int], col: int | np.ndarray[int]) -> int | np.ndarray[int, int]:
        """
        Convert 2D indices to a 1D index (both 0-based)
        """
        if isinstance(row, int) and isinstance(col, int):
            # Check if indices are within bounds
            if row > self.no_rows:
                raise ValueError(f'Row index {row} is out of bounds')
            if col > self.no_columns:
                raise ValueError(f'Column index {col} is out of bounds')
            # Calculate 1D index
            return row * self.no_columns + col
        elif isinstance(row, np.ndarray) and isinstance(col, np.ndarray):
            # Check if row and col are broadcastable
            if row.size != col.size:
                raise ValueError('Row and column indices must have the same length')
            # Check if indices are within bounds
            rows_out_of_range = np.where(row > self.no_rows)[0]
            if rows_out_of_range.size > 0:
                raise ValueError(f'Row indices {row[rows_out_of_range]} are out of bounds')
            cols_out_of_range = np.where(col > self.no_columns)[0]
            if cols_out_of_range.size > 0:
                raise ValueError(f'Column indices {col[cols_out_of_range]} are out of bounds')
            # Calculate 1D indices
            return row * self.no_columns + col
        else:
            raise TypeError(f'Incompatible types for \'row\' and \'col\': {type(row)} and {type(col)}')

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

    def _location_to_map(self, location: str | list) -> tuple[np.array, np.array]:
        # Parse location and get the indices of the valid positions
        indices = self._location_to_indices(location)

        # Initialize the flattened mask array with nan
        mask = np.full((self.size, ), np.nan)  # 1D array (size, )
        # Set values of the valid positions to 1.0
        mask[indices] = 1.0
        # Get the 2D indices of all valid positions
        mask_indices = self._index_1D_to_2D(np.where(mask == 1.0)[0])  # [[r1, c1], [r2, c2], ...]

        # Calculate the bounding box of the mask
        r_min, c_min = np.min(mask_indices, axis=0)  # [r_min, c_min]
        r_max, c_max = np.max(mask_indices, axis=0)  # [r_max, c_max]

        # Trim mask to the bounding box
        mask = mask.reshape(self.shape)  # reshape to 2D array (rows, columns)
        mask = mask[r_min:r_max+1, c_min:c_max+1]

        # Calculate the mapping of the location to the original experiment shape
        location_map = np.full(self.shape, False)
        location_map[r_min:r_max+1, c_min:c_max+1] = True

        return location_map, mask

    def _reshape_data(self, array: np.array, location_map: np.array, mask: np.array = None) -> tuple[np.array, np.array]:
        # Create dummy mask if not provided
        if isinstance(mask, NoneType):
            mask = np.ones_like(array)

        # Check if arrays are compatible
        if array.shape != mask.shape:
            raise ValueError('\'array\' and \'mask\' must have the same shape')
        if location_map.shape != self.shape:
            raise ValueError('\'location_map\' must have the same shape as the experiment')

        # Calculate the bounding box of the mask
        mask_indices = np.where(location_map == True)
        r_min, c_min = np.min(mask_indices, axis=1)  # [r_min, c_min]
        r_max, c_max = np.max(mask_indices, axis=1)  # [r_max, c_max]

        # Apply the mask to the input array
        masked_array = array * mask

        # Reshape the masked array to the experiment shape
        data_reshaped = np.full(self.shape, np.nan)
        data_reshaped[r_min:r_max+1, c_min:c_max+1] = masked_array

        # Reshape the mask to the experiment shape
        mask_reshaped = np.full(self.shape, np.nan)
        mask_reshaped[r_min:r_max+1, c_min:c_max+1] = mask
        return data_reshaped, mask_reshaped

    def apply_reagent(self, reagent: Reagent | ReagentWV | ReagentVV | Buffer,
                      applicator: ConstantApplicator | GradientApplicator | StepFixedApplicator,
                      location: str | list = 'everywhere', *,
                      pH_applicator: ConstantApplicator | GradientApplicator | StepFixedApplicator = None):
        # Type checking for reagent and applicator
        if not isinstance(reagent, _Reagent):
            raise TypeError('Invalid \'reagent\' type')
        if not isinstance(applicator, _ReagentApplicator):
            raise TypeError('Invalid \'applicator\' type')

        # Type checking for pH applicator
        do_pH_gradient = False
        if not isinstance(pH_applicator, NoneType):
            if not isinstance(pH_applicator, _ReagentApplicator):
                raise TypeError('Invalid \'pH_applicator\' type')
            if not isinstance(reagent, Buffer):
                raise TypeError('\'pH_applicator\' can only be applied to a Buffer reagent')
            do_pH_gradient = True

        # Check if the reagent is already in the list
        if reagent in self._reagents:
            warnings.warn(ExistingReagentWarning(raiser=reagent))
            return

        # Map the locations to the experiment shape
        #  loc_map: self.shape, mask.shape <= self.shape
        loc_map, mask = self._location_to_map(location)
        shape = mask.shape

        # Calculate reagent concentrations
        reagent.applicator = applicator
        data = reagent.applicator.apply(shape)
        data = data.reshape(shape)  # data.shape = mask.shape

        # Transform concentration array to the experiment shape
        #  data_reshaped: self.shape, mask_reshaped: self.shape
        data_reshaped, mask_reshaped = self._reshape_data(array=data, location_map=loc_map, mask=mask)

        # Flatten and mask array
        #  data_flattened: (self.size, )
        data_flattened = data_reshaped.ravel() * mask_reshaped.ravel()

        if not do_pH_gradient:  # for non-pH gradients
            # Append concentration data to the experiment
            if isinstance(self._data, NoneType):
                self._data = data_flattened.reshape((1, self.size))  # (1, size)
            else:
                self._data = np.vstack([self._data, data_flattened])  # (no_reagents, size)

            # Add reagent to the list
            self._reagents.append(reagent)
        else:  # for pH gradients
            reagent.pH_applicator = pH_applicator
            # Calculate pH values
            #  pH_data.shape = mask.shape
            pH_data = reagent.pH_applicator.apply(shape).reshape(shape)

            # Transform pH array to the experiment shape
            #  pH_reshaped: self.shape, pH_mask: self.shape
            pH_reshaped, pH_mask = self._reshape_data(array=pH_data, location_map=loc_map, mask=mask)

            # Flatten and mask pH array
            pH_flattened = pH_reshaped.ravel() * pH_mask.ravel()  # (size, )

            # Determine unique pH values
            unique_pHs = np.sort(np.unique(pH_flattened))

            # Create a new Buffer instance for each unique pH value
            for pH in unique_pHs:
                # Get indices for given pH
                indices = np.where(pH_flattened == pH)[0]

                # Set the concentration for this Buffer
                conc_data = np.full_like(data_flattened, np.nan)  # (size, )
                conc_data[indices] = data_flattened[indices]

                # Append concentration and pH data to experiment
                if isinstance(self._data, NoneType):
                    self._data = conc_data.reshape((1, self.size))
                else:
                    self._data = np.vstack([self._data, conc_data])
                self._pH[indices] = pH

                # Create new Buffer and add to reagent list
                buffer = Buffer(name=reagent.name, concentration=reagent.concentration, pH=pH)
                buffer.applicator = reagent.applicator  # ToDo: Convert this to ValuesApplicator
                buffer.pH_applicator = ConstantApplicator(value=pH)
                self._reagents.append(buffer)

    def calculate_volumes(self, final_volume: float | int):
        # V1 = C2 * V2 / C1
        c_stocks = np.array([reagent.concentration for reagent in self._reagents])
        v_stocks = ((self._data * final_volume).T / c_stocks).T
        v_water = final_volume - np.nansum(v_stocks, axis=0)

        impossibles = np.where(v_water < 0)[0]
        if impossibles.size > 0:
            raise ValueError(f'Impossible condition: Negative volume of water in well(s) {impossibles}')

        self._volumes = np.vstack((v_stocks, v_water))
        return self._volumes
