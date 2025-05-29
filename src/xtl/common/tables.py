from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import (List, Dict, Any, Optional, Union, Iterator, Iterable, Tuple,
                    Callable, Set)

from xtl.common import XTLUndefined


@dataclass
class MissingValue:
    """Represents a missing value.

    :param value: The original value
    """
    value: Any

    def __str__(self):
        return str(self.to_repr())

    def __repr__(self):
        return f'MissingValue({self.value!r})'


@dataclass
class MissingValueConfig:
    """Defines which values should be treated as missing and how they should be
    represented.

    :param values: Values to consider as missing
    :param repr: Value to use when representing missing values
    :param checker: Optional function to check if a value should be considered missing
    """
    values: Set[Any] = field(default_factory=set)
    repr: Any = None
    checker: Optional[Callable[[Any], bool]] = None

    def __init__(self, values: Any, repr: Any = XTLUndefined,
                 checker: Optional[Callable[[Any], bool]] = None):
        # Process values
        if values == XTLUndefined:
            # Create an empty set if no values are provided
            self.values = set()
        elif isinstance(values, (list, tuple, set)):
            # Convert interables to set
            self.values = set(values)
        else:
            # Single value
            self.values = {values}

        # Set the representation
        if repr == XTLUndefined:
            self.repr = None
        else:
            self.repr = repr

        # Set the checker function
        self.checker = checker

    def __contains__(self, value: Any) -> bool:
        """Check if a value is considered missing.

        :param value: The value to check
        :return: True if the value is considered missing, False otherwise
        """
        if value in self.values:
            return True
        elif self.checker is not None:
            return self.checker(value)
        return False

    def to_repr(self, value):
        """Convert a value to its representation if it is considered missing.

        :param value: The value to convert
        :return: The representation if the value is missing, otherwise the original value
        """
        if isinstance(value, MissingValue) or value in self:
            return self.repr
        return value


class Table:
    def __init__(self, data: Optional[List[List[Any]]] = None, *,
                 headers: Optional[List[str]] = None,
                 missing_values: MissingValueConfig | Any = XTLUndefined,
                 missing_value_repr: Any = None):
        """Initialize a Table object.

        :param data: Optional data as a list of rows
        :param headers: Optional list of column names
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        """
        self._headers = list(headers) if headers is not None else []
        self._data = []

        # Handle missing values
        self.default_missing_value: Any = None
        self._missing: MissingValueConfig
        self._set_missing(value=missing_values, value_repr=missing_value_repr)

        # Process initial data if provided
        if data:
            for row in data:
                self.add_row([self._process_value(v) for v in row])

    def _process_value(self, value: Any) -> Any:
        """Check if a value is considered missing."""
        if isinstance(value, MissingValue):
            return value
        elif value in self._missing:
            return MissingValue(value=value)
        return value

    def _set_missing(self, value: Any, value_repr: Any | Iterable[Any] = XTLUndefined):
        """Set the missing value configuration.

        :param value: Value(s) to consider as missing
        :param value_repr: Representation for missing values
        """
        if isinstance(value, MissingValueConfig):
            self._missing = value
        else:
            self._missing = MissingValueConfig(values=value, repr=value_repr)

    @property
    def missing(self) -> MissingValueConfig:
        """Get the MissingValueConfig instance for this table.

        :return: The MissingValueConfig instance
        """
        return self._missing

    @missing.setter
    def missing(self, value: Union[MissingValueConfig, Any]):
        """Set the missing value configuration for this table.

        :param value: MissingValueConfig instance or value(s) to consider missing
        """
        current_repr = XTLUndefined
        if hasattr(self, '_missing') and self._missing is not None:
            current_repr = self._missing.repr

        if isinstance(value, MissingValueConfig):
            # If a MissingValueConfig instance is provided, it carries its own repr.
            self._set_missing(value=value)
        else:
            # If a list/single value is provided for missing values,
            # reuse the current representation.
            self._set_missing(value=value, value_repr=current_repr)

        # Update existing data with new missing value definitions
        for i, row in enumerate(self._data):
            self._data[i] = [self._update_value(v) for v in row]

    def _update_value(self, value: Any) -> Any:
        """Process a value when the missing value configuration changes."""
        if isinstance(value, MissingValue):
            original_value = value.value
            if original_value in self._missing:
                return MissingValue(value=original_value)
            return original_value
        return self._process_value(value)

    def _to_repr(self, value: Any) -> Any:
        """Convert a value for output."""
        if isinstance(value, MissingValue):
            return self._missing.to_repr(value)
        return value

    @property
    def headers(self) -> List[str]:
        """Get the column headers.

        :return: The column headers
        """
        return self._headers

    @headers.setter
    def headers(self, headers: List[str]):
        """Set the column headers for the table.

        :param headers: List of column names to set as headers
        :raises ValueError: If the number of columns does not match the width of existing rows
        """
        if len(self._data) > 0 and len(headers) != len(self._data[0]):
            raise ValueError('Number of headers does not match row width')
        self._headers = list(headers)

    @property
    def no_rows(self) -> int:
        """Get the number of rows in the table.

        :return: The number of rows
        """
        return len(self._data)

    @property
    def no_cols(self) -> int:
        """Get the number of columns in the table.

        :return: The number of columns
        """
        if self._headers:
            return len(self._headers)
        if self._data:
            return len(self._data[0])
        return 0

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the table as (rows, columns).

        :return: A tuple with the number of rows and columns
        """
        return self.no_rows, self.no_cols

    @property
    def size(self) -> int:
        """Get the total number of elements in the table.

        :return: The total number of elements (rows * columns)
        """
        return self.no_rows * self.no_cols

    @property
    def has_missing(self) -> bool:
        """Check if the table contains any missing values.

        :return: True if there are missing values, False otherwise
        """
        for row in self._data:
            if any(isinstance(v, MissingValue) for v in row):
                return True
        return False

    def add_row(self, data: List[Any] | Dict[Union[str, int], Any]):
        """Append a new row to the table.

        :param data: The row data to append. Can be either:
            - A list of values in column order
            - A dictionary mapping column names/indices to values
        :raises ValueError: If the row length does not match the number of columns
        :raises KeyError: If a dictionary key is a column name that doesn't exist
        :raises IndexError: If a dictionary key is a column index that is out of range
        """
        # Handle dictionary input
        if isinstance(data, dict):
            dummy_missing = 'XTLDummyMissingValue'
            # Prepare a row with missing values
            new_row = [dummy_missing] * self.no_cols

            # Fill in the values from the dictionary
            for key, value in data.items():
                if isinstance(key, str):
                    # Handle column name
                    if key not in self._headers:
                        raise KeyError(f'Column name {key!r} does not exist')
                    col_idx = self._headers.index(key)
                else:
                    # Handle column index
                    col_idx = key
                    # Handle negative indices
                    if col_idx < 0:
                        col_idx += self.no_cols
                    # Check if index is valid
                    if col_idx < 0 or col_idx >= self.no_cols:
                        raise IndexError(f'Column index {key} out of range')

                new_row[col_idx] = value

            # Process the row (convert to MissingValue objects where appropriate)
            processed_row = [self._process_value(v) if v != dummy_missing
                             else MissingValue(self.default_missing_value)
                             for v in new_row]
            self._data.append(processed_row)

        # Handle iterable input
        else:
            if self._data and len(data) != self.no_cols:
                raise ValueError('Row length does not match headers')

            processed_row = [self._process_value(v) for v in data]
            self._data.append(processed_row)

    def add_col(self, data: List[Any] | Dict[str, Any] | str = None, *,
                col_name: Optional[str] = None):
        """Append a new column to the table.

        :param data: The column data to append
        :param col_name: The name of the new column
        :raises ValueError: If the number of values does not match the number of rows
        :raises ValueError: If the table has headers but no column name is provided
        """
        if isinstance(data, str):
            # Handle creation of empty columns
            col_name = data
            data = None
        elif isinstance(data, dict):
            # Handle dict input
            if len(data.keys()) > 1:
                raise ValueError('Only one column can be added at a time with a dict '
                                 'input')
            if col_name is None:
                col_name = list(data.keys())[0]
            data = list(data.values())[0]

        # If no data is provided, create a column of missing values
        dummy_missing = 'XTLDummyMissingValue'
        if data is None:
            data = [dummy_missing] * self.no_rows

        # Handle column name
        if col_name is None:
            if self._headers:
                raise ValueError('Column name must be provided when table has headers')
        else:
            self._headers.append(col_name)

        if self._data and len(data) != self.no_rows:
            raise ValueError('Column length does not match number of rows')

        processed_values = [self._process_value(v) if v != dummy_missing
                            else MissingValue(self.default_missing_value)
                            for v in data]
        for i, val in enumerate(processed_values):
            self._data[i].append(val)

    def get_row(self, idx: int) -> List[Any]:
        """Get a row by index.

        :param idx: The row index
        """
        return [self._to_repr(val) for val in self._data[idx]]

    def get_col(self, idx: Union[str, int]) -> List[Any]:
        """Get a column by name or index.

        :param idx: The column name or index
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        """
        if isinstance(idx, str):
            if idx not in self._headers:
                raise KeyError(f'Column {idx!r} does not exist')
            col_idx = self._headers.index(idx)
        else:
            col_idx = idx
            # Handle negative indices (e.g., -1 for the last column)
            if col_idx < 0 and self._data:
                col_idx += len(self._data[0])  # Convert negative index to positive

            # Check if index is valid
            if col_idx < 0 or (self._data and col_idx >= len(self._data[0])) or \
                    (not self._data and col_idx > 0):
                raise IndexError(f'Column index {col_idx} out of range')

        # Process output values to handle missing values
        return [self._to_repr(row[col_idx]) for row in self._data]

    def set_row(self, idx: int, row: List[Any]):
        """Set the data for a row at a given index.

        :param idx: The row index
        :param row: The new row data
        :raises ValueError: If the row length does not match the number of columns
        """
        if len(row) != len(self._headers):
            raise ValueError('Row length does not match headers')

        self._data[idx] = [self._process_value(val) for val in row]

    def set_col(self, idx: Union[str, int], values: List[Any]):
        """Set the data for a column by name or index.

        :param idx: The column name or index
        :param values: The new column data
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        :raises ValueError: If the number of values does not match the number of rows
        """
        if len(values) != len(self._data):
            raise ValueError('Column length does not match number of rows')

        if isinstance(idx, str):
            if idx not in self._headers:
                raise KeyError(f'Column {idx} does not exist')
            col_idx = self._headers.index(idx)
        else:
            col_idx = idx
            if col_idx < 0 or (self._data and col_idx >= len(self._data[0])) or \
                    (not self._data and col_idx > 0):
                raise IndexError(f'Column index {col_idx} is out of range')

        # Process input values to handle missing values
        processed_values = [self._process_value(val) for val in values]
        for i, val in enumerate(processed_values):
            self._data[i][col_idx] = val

    @property
    def data(self) -> List[List[Any]]:
        """Get the raw data of the table.

        :return: The table data as a list of rows
        """
        return [[self._to_repr(v) for v in row] for row in self._data]

    def to_list(self) -> List[Any]:
        """Convert the table to a flattened list

        :return: The table data as a list
        """
        return [self._to_repr(v) for row in self._data for v in row]

    def to_dict(self) -> Dict[str | int, List[Any]]:
        """Convert the table to a dictionary with column names as keys.

        :return: The table data as a dictionary
        """
        headers = self._headers if self._headers else list(range(len(self._data[0])))
        print(headers)

        return {col: self.get_col(col) for col in headers}

    def to_numpy(self) -> 'numpy.ndarray':
        """Convert the table to a numpy ndarray. Requires numpy to be installed.

        :return: The table as a numpy array
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise ImportError('numpy is not installed')

        return np.array(self.data)

    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert the table to a pandas DataFrame. Requires pandas to be installed.

        :return: The table as a DataFrame
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ImportError('pandas is not installed')

        return pd.DataFrame(self.data, columns=self._headers if self._headers else None)

    def to_rich(self, cast_as: Callable = None) -> 'rich.table.Table':
        """Convert the table to a rich.Table for pretty console output. Requires rich to
        be installed.

        :param cast_as: Optional function to cast values before adding to the rich table
        :return: The table as a rich Table object
        """
        try:
            from rich.table import Table as RichTable
        except ModuleNotFoundError:
            raise ImportError('rich is not installed')

        table = RichTable()
        if self._headers:
            for col in self._headers:
                table.add_column(col)

        for row in self.data:
            if cast_as:
                values = [cast_as(v) for v in row]
            else:
                values = [str(v) if v else None for v in row]
            table.add_row(*values)
        return table

    def to_csv(self, filename: str | Path = None, delimiter: str = ',',
               new_line: str = '\n', header_char: str = '', overwrite: bool = False,
               keep_file_ext: bool = False) -> str | Path:
        """Write the table to a CSV file. If ``filename`` is not provided, then the
        CSV will be returned as a string.

        :param filename: Optional output path for the CSV file.
        :param delimiter: Delimiter to use in the CSV.
        :param new_line: Newline character to use in the CSV.
        :param header_char: Character to prepend to the header line (e.g., '#').
        :param overwrite: Overwrite the file if it already exists.
        :param keep_file_ext: Keep the file extension if ``filename`` is provided.
        :return: Either the CSV string or the output path.
        :raises FileExistsError: If the file already exists and ``overwrite`` is False
        """
        result = ''
        delimiter_len = len(delimiter)

        if self._headers:
            # Add headers to the CSV
            result += header_char + delimiter.join(self._headers) + new_line

        for row in self._data:
            for val in row:
                result += str(self._to_repr(val)) + delimiter
            result = result[:-delimiter_len]  # Remove trailing delimiter
            result += new_line

        # If no filename is provided, then return a CSV string
        if filename is None:
            return result

        # If a filename is provided, then write to a file
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File already exists: {filename}')
        if filename.suffix != '.csv' and not keep_file_ext:
            filename = filename.with_suffix('.csv')

        filename.write_text(result)
        return filename

    @classmethod
    def from_dict(cls, data: Dict[str | int, List[Any]], *,
                  missing_values: MissingValueConfig | Any = XTLUndefined,
                  missing_value_repr: Any = None) -> 'Table':
        """Create a Table from a dictionary of columns.

        :param data: Dictionary with column names as keys and column data as values
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        :return: A new Table instance
        :raises ValueError: If the columns are not of equal length
        """
        if not data:
            return cls(headers=[], missing_values=missing_values,
                       missing_value_repr=missing_value_repr)

        # Extract headers and check column lengths
        headers = list(data.keys())
        column_lengths = [len(col) for col in data.values()]

        if len(set(column_lengths)) > 1:
            raise ValueError('All columns must have the same length')

        num_rows = column_lengths[0] if column_lengths else 0

        # Convert dict of columns to list of rows
        rows = []
        for i in range(num_rows):
            row = [data[col][i] for col in headers]
            rows.append(row)

        return cls(data=rows, headers=headers, missing_values=missing_values,
                   missing_value_repr=missing_value_repr)

    @classmethod
    def from_numpy(cls, array: 'np.ndarray', *,
                   headers: Optional[List[str]] = None,
                   missing_values: MissingValueConfig | Any = XTLUndefined,
                   missing_value_repr: Any = None) -> 'Table':
        """Create a Table from a numpy ndarray.

        :param array: 2D numpy array
        :param headers: Optional list of column names
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        :return: A new Table instance
        :raises ImportError: If numpy is not installed
        :raises ValueError: If the array is not 2D
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise ImportError('numpy is not installed')

        if len(array.shape) != 2:
            raise ValueError('Array must be 2D')

        data = array.tolist()

        return cls(data=data, headers=headers, missing_values=MissingValueConfig(
            values=missing_values, repr=missing_value_repr, checker=lambda v: v is np.nan
        ))

    @classmethod
    def from_pandas(cls, df: 'pd.DataFrame', *,
                    missing_values: MissingValueConfig | Any = XTLUndefined,
                    missing_value_repr: Any = None) -> 'Table':
        """Create a Table from a pandas DataFrame.

        :param df: pandas DataFrame
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        :return: A new Table instance
        :raises ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ImportError('pandas is not installed')

        headers = df.columns.tolist()
        data = df.values.tolist()

        return cls(data=data, headers=headers, missing_values=MissingValueConfig(
            values=missing_values, repr=missing_value_repr, checker=lambda v: pd.isna(v)
        ))

    @classmethod
    def from_csv(cls, s: str | Path, *, delimiter: str = ',', new_line: str = '\n',
                 header_line: int = None, header_char: str = '') -> 'Table':
        """Create a Table from a CSV file or string.

        :param s: Path to CSV file or CSV string content
        :param delimiter: Delimiter used in the CSV (default: ',')
        :param header_line: Line that contains headers (default: None)
        :param header_char: Character that might prefix the header line (e.g., '#')
        :return: A new Table instance
        """
        # Check if the input is empty
        if not s:
            return cls(headers=[])

        # Check if s is a file path or string content
        is_file = isinstance(s, (str, Path)) and Path(s).exists()

        if is_file:
            # Read from file
            content = Path(s).read_text()
        else:
            # Use source as raw CSV content
            content = s

        # Split into lines and process
        lines = content.split(new_line)
        if not lines:
            return cls(headers=[])

        # Process headers
        headers = None
        data_idx = 0

        if header_line is not None:
            data_idx = header_line + 1
            header_line = lines[header_line]
            # Remove header character if present
            if header_char and header_line.startswith(header_char):
                header_line = header_line[len(header_char):]
            headers = [h for h in header_line.split(delimiter)]

        # Process data rows
        data = []
        for i in range(data_idx, len(lines)):
            line = lines[i]
            # Skip empty lines
            if not line:
                continue

            # Split by delimiter
            data.append(line.split(delimiter))

        return cls(data=data, headers=headers)

    def __str__(self):
        """Return a pretty-printed string representation of the table.

        :return: String representation of the table
        """
        if not self._data:
            return '(Empty table)'

        # Process output values to handle missing values
        data = self.data

        if not self._headers:
            # Print as plain rows
            return '\n'.join(' | '.join(str(x) for x in row) for row in data)

        col_widths = [max(len(str(col)), max((len(str(row[i])) for row in data), default=0)) for i, col in enumerate(self._headers)]
        header = ' | '.join(col.ljust(col_widths[i]) for i, col in enumerate(self._headers))
        sep = '-+-'.join('-' * w for w in col_widths)
        rows = [' | '.join(str(row[i]).ljust(col_widths[i]) for i in range(len(self._headers))) for row in data]
        return '\n'.join([header, sep] + rows)

    def __len__(self):
        """Return the number of rows in the table."""
        return self.no_rows

    def __getitem__(self, key):
        """Get data from the table using pandas-like slicing syntax.

        Supports:
        - table[col] -> Single column (by name or index)
        - table[col0:col1] -> New Table with selected columns
        - table[col, row] -> Single cell value
        - table[col, row0:row1] -> New Table with selected part of a column
        - table[col0:col1, row0:row1] -> New Table with subset of the table

        :param key: Index, slice, string, or tuple for advanced indexing
        :return: Single value, list (single column), or Table instance (multiple values)
        :raises IndexError: If indices are out of range
        :raises KeyError: If column names don't exist
        """
        # Case 1: table[col, row] or table[col, row_slice]
        if isinstance(key, tuple):
            col_spec, row_spec = key

            # Get the columns first
            if isinstance(col_spec, slice):
                # Handle column slicing: table[col0:col1, ...]
                col_indices = self._resolve_column_slice(col_spec)
                columns = [self.get_col(i) for i in col_indices]
                col_headers = [self._headers[i] for i in col_indices]
            else:
                # Handle single column: table[col, ...]
                try:
                    columns = [self.get_col(col_spec)]
                    if isinstance(col_spec, int):
                        col_headers = [self._headers[col_spec]]
                    else:
                        col_headers = [col_spec]
                except (KeyError, IndexError) as e:
                    raise e

            # Now handle the row specification
            if isinstance(row_spec, slice):
                # Handle row slicing: table[..., row0:row1]
                row_indices = self._resolve_row_slice(row_spec)

                # For each column, extract the row slice
                result = []
                for col in columns:
                    result.append([col[i] for i in row_indices])

                # If only one column was requested, create a single-column table
                if len(result) == 1:
                    # Create a new table with the column data
                    new_data = [[val] for val in result[0]]
                    return Table(data=new_data, headers=col_headers,
                                 missing_values=self._missing)

                # Otherwise, transpose the result to get rows and create a new table
                new_data = [[result[c][r] for c in range(len(result))]
                            for r in range(len(result[0]))]
                return Table(data=new_data, headers=col_headers,
                             missing_values=self._missing)
            else:
                # Handle single row: table[..., row]
                if not isinstance(row_spec, int):
                    raise TypeError(f'Row indices must be integers or slices, not '
                                    f'{type(row_spec).__name__}')

                # Adjust negative indices
                row_idx = row_spec
                if row_idx < 0:
                    row_idx += len(self._data)

                # Check if index is valid
                if row_idx < 0 or row_idx >= len(self._data):
                    raise IndexError(f'Row index {row_spec} is out of bounds')

                # If only one column was requested, return the single value
                if len(columns) == 1:
                    return columns[0][row_idx]

                # Otherwise, return a new table with the single row
                new_data = [[col[row_idx] for col in columns]]
                return Table(data=new_data, headers=col_headers,
                             missing_values=self._missing)

        # Case 2: table[col] or table[col0:col1]
        elif isinstance(key, (str, int, slice)):
            if isinstance(key, slice):
                # Handle column slicing: table[col0:col1]
                col_indices = self._resolve_column_slice(key)
                col_headers = [self._headers[i] for i in col_indices]

                # Get each column and transpose to rows
                columns = [self.get_col(i) for i in col_indices]
                if not columns:
                    # Empty table with selected headers
                    return Table(headers=col_headers, missing_values=self._missing)

                new_data = [[columns[c][r] for c in range(len(columns))]
                            for r in range(len(columns[0]))]
                return Table(data=new_data, headers=col_headers,
                             missing_values=self._missing)
            else:
                # Handle single column: table[col]
                try:
                    column_data = self.get_col(key)
                    # Create and return a new table with this single column
                    if isinstance(key, int):
                        col_header = [self._headers[key]]
                    else:
                        col_header = [key]
                    # Create the data as a list of single-item rows
                    new_data = [[val] for val in column_data]
                    return Table(data=new_data, headers=col_header,
                                 missing_values=self._missing)
                except (KeyError, IndexError) as e:
                    raise e

        else:
            raise TypeError(f'Invalid key type: {type(key).__name__}')

    def _resolve_column_slice(self, col_slice: slice) -> List[int]:
        """Resolve a column slice to a list of column indices.

        :param col_slice: The slice object for columns
        :return: List of column indices
        """
        # Determine start, stop, and step
        start = col_slice.start
        stop = col_slice.stop
        step = col_slice.step if col_slice.step is not None else 1

        # If start or stop are strings, convert to indices
        if isinstance(start, str):
            try:
                start = self._headers.index(start)
            except ValueError:
                raise KeyError(f'Column {start!r} does not exist')

        if isinstance(stop, str):
            try:
                stop = self._headers.index(stop) + 1  # Include the stop column
            except ValueError:
                raise KeyError(f'Column {stop!r} does not exist')

        # Handle None values in slice
        num_cols = len(self._headers)
        if start is None:
            start = 0
        elif start < 0:
            start += num_cols

        if stop is None:
            stop = num_cols
        elif stop < 0:
            stop += num_cols

        # Generate indices and validate
        indices = list(range(start, stop, step))
        if indices and (indices[0] < 0 or indices[-1] >= num_cols):
            raise IndexError(f'Column slice {col_slice} out of bounds')

        return indices

    def _resolve_row_slice(self, row_slice: slice) -> List[int]:
        """Resolve a row slice to a list of row indices.

        :param row_slice: The slice object for rows
        :return: List of row indices
        """
        # Determine start, stop, and step
        start = row_slice.start if row_slice.start is not None else 0
        stop = row_slice.stop if row_slice.stop is not None else len(self._data)
        step = row_slice.step if row_slice.step is not None else 1

        # Handle negative indices
        if start < 0:
            start += len(self._data)
        if stop < 0:
            stop += len(self._data)

        # Generate indices and validate
        indices = list(range(start, stop, step))
        if indices and (indices[0] < 0 or indices[-1] >= len(self._data)):
            raise IndexError(f'Row slice {row_slice} out of bounds')

        return indices

    def __setitem__(self, key, value):
        """Set data in the table using pandas-like slicing syntax.

        Supports:
        - table[col] = values -> Set an entire column
        - table[col, row] = value -> Set a single cell
        - table[col, row0:row1] = values -> Set part of a column

        :param key: Index, slice, string, or tuple for advanced indexing
        :param value: The value(s) to set
        :raises IndexError: If indices are out of range
        :raises KeyError: If column names don't exist
        :raises ValueError: If value dimensions don't match the target
        """
        # Case 1: table[col, row] = value or table[col, row_slice] = values
        if isinstance(key, tuple):
            col_spec, row_spec = key

            # Handle the column specification
            if isinstance(col_spec, slice):
                raise NotImplementedError('Setting multiple columns with a slice is '
                                          'not supported')

            # Get the current column data
            try:
                col_data = self.get_col(col_spec)
            except (KeyError, IndexError) as e:
                raise e

            # Now handle the row specification
            if isinstance(row_spec, slice):
                # Handle row slicing: table[col, row0:row1] = values
                row_indices = self._resolve_row_slice(row_spec)

                # Validate the value length
                if not hasattr(value, '__len__'):
                    # Scalar value - repeat for all indices
                    value = [value] * len(row_indices)
                elif len(value) != len(row_indices):
                    raise ValueError(f'Value length {len(value)} does not match target '
                                     f'length {len(row_indices)}')

                # Update the column data
                new_col_data = list(col_data)
                for i, idx in enumerate(row_indices):
                    new_col_data[idx] = value[i]

                # Set the updated column
                self.set_col(col_spec, new_col_data)
            else:
                # Handle single row: table[col, row] = value
                if not isinstance(row_spec, int):
                    raise TypeError(f'Row indices must be integers or slices, not '
                                    f'{type(row_spec).__name__}')

                # Adjust negative indices
                row_idx = row_spec
                if row_idx < 0:
                    row_idx += len(self._data)

                # Check if index is valid
                if row_idx < 0 or row_idx >= len(self._data):
                    raise IndexError(f'Row index {row_spec} is out of bounds')

                # Update the single value
                new_col_data = list(col_data)
                new_col_data[row_idx] = value

                # Set the updated column
                self.set_col(col_spec, new_col_data)

        # Case 2: table[col] = values
        elif isinstance(key, (str, int)):
            # Handle setting an entire column
            try:
                self.set_col(key, value)
            except (KeyError, IndexError, ValueError) as e:
                raise e

        # Case 3: table[col0:col1] = values (not supported)
        elif isinstance(key, slice):
            raise NotImplementedError('Setting multiple columns with a slice is not '
                                      'supported')

        else:
            raise TypeError(f'Invalid key type: {type(key).__name__}')

    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over rows in the table."""
        for row in self._data:
            yield [self._to_repr(val) for val in row]

    def items(self) -> Iterator[Tuple[str, List[Any]]]:
        """Iterate over columns as (name, values) pairs."""
        for col_name in self._headers:
            yield (col_name, self.get_col(col_name))

    def __add__(self, other: 'Table') -> 'Table':
        """Concatenate rows from another table to this table.

        The two tables must have the same column structure (headers).

        :param other: The table to append rows from
        :return: A new table with combined rows
        :raises ValueError: If the tables have different column headers
        """
        if not isinstance(other, Table):
            raise TypeError(f'Cannot add Table and {type(other).__name__}')

        # Check if both tables have the same columns
        if self._headers != other._headers:
            raise ValueError('Tables must have the same column headers to be '
                             'concatenated')

        # Create a new table with combined data
        combined_data = [row for row in self._data]  # Create a copy of the data
        combined_data.extend(other._data)

        # Use the same missing value configuration
        return Table(data=combined_data, headers=self._headers[:],
                     missing_values=self._missing)

    def __or__(self, other: 'Table') -> 'Table':
        """Concatenate columns from another table to this table.

        The two tables must have the same number of rows.

        :param other: The table to append columns from
        :return: A new table with combined columns
        :raises ValueError: If the tables have different numbers of rows
        """
        if not isinstance(other, Table):
            raise TypeError(f'Cannot use \'|\' operator between Table and '
                            f'{type(other).__name__}')

        # Check if both tables have the same number of rows
        if len(self) != len(other):
            raise ValueError('Tables must have the same number of rows to concatenate '
                             'columns')

        # Combine headers
        combined_headers = self._headers + other._headers

        # Create the combined data
        combined_data = []
        for i in range(len(self)):
            combined_row = self._data[i] + other._data[i]
            combined_data.append(combined_row)

        # Create a new table with combined data
        # For missing values, prefer the configuration from the first table
        return Table(data=combined_data, headers=combined_headers,
                     missing_values=self._missing)

    def __iadd__(self, other: Union['Table', List[Any], Dict[Union[str, int], Any]]) -> 'Table':
        """In-place row concatenation (append rows from another table or add a new row).

        The method can be used in two ways:
        1. Append rows from another table (tables must have the same column structure)
        2. Add a single row as a list or dictionary

        :param other: The table to append rows from or the row data to append
        :return: Self with rows from other table appended or with new row added
        :raises ValueError: If the tables have different column headers
        """
        if isinstance(other, Table):
            if other is self:
                # Guard against self-reference which would cause infinite recursion
                headers = deepcopy(other._headers)
                data = deepcopy(other._data)
            else:
                headers = other._headers
                data = other._data

            # Check if both tables have the same columns
            if self._headers != headers:
                raise ValueError('Tables must have the same column headers to be '
                                 'concatenated')

            # Append the data from other table to this table
            for row in data:
                self._data.append(row)
        else:
            # New behavior: Add a single row
            self.add_row(other)

        # Return self for chaining operations
        return self

    def __ior__(self, other: Union['Table', List[Any], Dict[str, Any], str]) -> 'Table':
        """In-place column concatenation (append columns from another table or add a new column).

        The method can be used in two ways:
        1. Append columns from another table (tables must have the same number of rows)
        2. Add a single column as a list, dictionary, or string (column name with default missing values)

        :param other: The table to append columns from, column data to append, or column name
        :return: Self with columns from other table appended or with new column added
        :raises ValueError: If the tables have different numbers of rows
        """
        if isinstance(other, Table):
            if other is self:
                # Guard against self-reference which would cause infinite recursion
                headers = deepcopy(other._headers)
                data = deepcopy(other._data)
            else:
                headers = other._headers
                data = other._data

            # Check if both tables have the same number of rows
            if len(self) != len(data):
                raise ValueError('Tables must have the same number of rows to concatenate '
                                 'columns')

            # Append the headers from other table
            self._headers.extend(headers)

            # Append the data from other table to this table's rows
            for i in range(self.no_rows):
                self._data[i].extend(data[i])
        else:
            # New behavior: Add a single column
            self.add_col(other)

        # Return self for chaining operations
        return self

    def del_row(self, idx: int):
        """Remove a row at the given index.

        :param idx: The row index to remove
        :raises IndexError: If the row index is out of range
        """
        # Handle negative indices
        row_idx = idx
        if row_idx < 0:
            row_idx += len(self._data)

        # Check if index is valid
        if row_idx < 0 or row_idx >= len(self._data):
            raise IndexError(f'Row index {idx} is out of bounds')

        # Delete the row
        self._data.pop(row_idx)

    def del_col(self, idx: Union[str, int]):
        """Remove a column by name or index.

        :param idx: The column name or index to remove
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        """
        # Convert column name to index if necessary
        if isinstance(idx, str):
            if idx not in self._headers:
                raise KeyError(f'Column {idx!r} does not exist')
            col_idx = self._headers.index(idx)
        else:
            col_idx = idx
            # Handle negative indices
            if col_idx < 0:
                col_idx += self.no_cols

            # Check if index is valid
            if col_idx < 0 or col_idx >= self.no_cols:
                raise IndexError(f'Column index {idx} is out of bounds')

        # Remove the column from each row
        for i in range(len(self._data)):
            self._data[i].pop(col_idx)

        # Remove the header if it exists
        if self._headers and col_idx < len(self._headers):
            self._headers.pop(col_idx)

    def __sub__(self, other: Union['Table', str, List[str], Tuple[str, ...]]) -> 'Table':
        """Remove columns from the table or compute the column-wise difference with another table.

        This method can be used in two ways:
        1. table - other_table: Return a new table with only the columns that are in this table
           but not in the other table (column-wise difference)
        2. table - 'col_name': Return a new table with the specified column removed
        3. table - ['col1', 'col2']: Return a new table with multiple columns removed

        :param other: The table to compute difference with, or column(s) to remove
        :return: A new table with columns removed
        :raises KeyError: If a column name does not exist
        """
        # Case 1: Remove columns by name
        if isinstance(other, (str, list, tuple)):
            # Convert single column name to list
            cols_to_remove = [other] if isinstance(other, str) else other

            # Verify all columns exist
            for col in cols_to_remove:
                if col not in self._headers:
                    raise KeyError(f'Column {col!r} does not exist')

            # Get indices of columns to keep
            keep_indices = [i for i, col in enumerate(self._headers)
                           if col not in cols_to_remove]

            # Create new headers and data
            new_headers = [self._headers[i] for i in keep_indices]
            new_data = []
            for row in self._data:
                new_data.append([row[i] for i in keep_indices])

            # Create and return new table
            return Table(data=new_data, headers=new_headers, missing_values=self._missing)

        # Case 2: Column-wise difference with another table
        elif isinstance(other, Table):
            # Get columns that are in self but not in other
            diff_cols = [col for col in self._headers if col not in other._headers]

            # If no columns differ, return an empty table with the same missing value config
            if not diff_cols:
                return Table(headers=[], missing_values=self._missing)

            # Get indices of columns to keep
            keep_indices = [self._headers.index(col) for col in diff_cols]

            # Create new data
            new_data = []
            for row in self._data:
                new_data.append([row[i] for i in keep_indices])

            # Create and return new table
            return Table(data=new_data, headers=diff_cols, missing_values=self._missing)

        else:
            raise TypeError(f'Cannot subtract {type(other).__name__} from Table')

    def __isub__(self, other: Union['Table', str, List[str], Tuple[str, ...]]) -> 'Table':
        """In-place removal of columns from the table.

        This method can be used in two ways:
        1. table -= other_table: Remove columns from this table that are also in the other table
        2. table -= 'col_name': Remove the specified column from this table
        3. table -= ['col1', 'col2']: Remove multiple columns from this table

        :param other: The table to compute difference with, or column(s) to remove
        :return: Self with columns removed
        :raises KeyError: If a column name does not exist
        """
        # Case 1: Remove columns by name
        if isinstance(other, (str, list, tuple)):
            # Convert single column name to list
            cols_to_remove = [other] if isinstance(other, str) else other

            # Remove columns one by one (in reverse order to avoid index shifting)
            for col in sorted([self._headers.index(col) for col in cols_to_remove], reverse=True):
                self.del_col(col)

        # Case 2: Column-wise difference with another table
        elif isinstance(other, Table):
            # Get columns that are in both tables
            common_cols = [col for col in self._headers if col in other._headers]

            # Remove common columns one by one (in reverse order to avoid index shifting)
            for col in sorted([self._headers.index(col) for col in common_cols], reverse=True):
                self.del_col(col)

        else:
            raise TypeError(f'Cannot subtract {type(other).__name__} from Table')

        return self
