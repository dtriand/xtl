from collections.abc import Mapping
import csv
from dataclasses import dataclass
import io
from numbers import Number
import operator
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Optional, Union

from xtl_core.system.compatibility.py313 import path_is_file
from xtl_core.system.compatibility.py314 import sentinel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import rich.table

# numpy, pandas and rich are optional dependencies, in an extra of xtl-core (name not final)
_EXTRA_HINT = 'install them with `pip install xtl-core[extra]`'

_EMPTY = sentinel('_EMPTY')
"""A cell that :meth:`Table.add_row` or :meth:`Table.add_col` was given no value for."""


def _is_nan(value: Any) -> bool:
    """Whether a value is a NaN number (of any type, ``float('nan')`` or ``np.nan`` alike)."""
    return isinstance(value, Number) and value != value


@dataclass
class MissingValue:
    """Represents a missing value.

    :param value: The original value
    """

    value: Any

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'MissingValue({self.value!r})'


class MissingValueConfig:
    """Defines which values should be treated as missing and how they should be
    represented.

    :param values: Value(s) to consider as missing: a list, tuple or set of values, or a
        single value. NaN matches any NaN.
    :param repr: Value to use when representing missing values
    :param checker: Optional function to check if a value should be considered missing
    """

    def __init__(self, values: Any = (), repr: Any = None, checker: Optional[Callable[[Any], bool]] = None):
        self.values: set[Any] = set(values) if isinstance(values, (list, tuple, set, frozenset)) else {values}
        self.repr = repr
        self.checker = checker

    def __repr__(self):
        return f'MissingValueConfig(values={self.values!r}, repr={self.repr!r}, checker={self.checker!r})'

    def __eq__(self, other):
        if not isinstance(other, MissingValueConfig):
            return NotImplemented
        return (self.values, self.repr, self.checker) == (other.values, other.repr, other.checker)

    __hash__ = None  # pyright: ignore[reportAssignmentType]  # mutable, so unhashable

    def __contains__(self, value: Any) -> bool:
        """Check if a value is considered missing.

        :param value: The value to check
        :return: True if the value is considered missing, False otherwise
        """
        try:
            if value in self.values:
                return True
        except TypeError:
            # Unhashable values (lists, arrays, ...) can only be missing through the checker
            pass
        if _is_nan(value) and any(_is_nan(v) for v in self.values):
            return True
        if self.checker is not None:
            return bool(self.checker(value))
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
    def __init__(
        self,
        data: Optional[Iterable[Iterable[Any]]] = None,
        *,
        headers: Optional[list[str]] = None,
        missing_values: Union[MissingValueConfig, Any] = (),
        missing_value_repr: Any = None,
        formats: Optional[dict[Union[str, int], str]] = None,
    ):
        """Initialize a Table object.

        :param data: Optional data as rows (e.g. a list of lists, or a 2D numpy array)
        :param headers: Optional list of column names. Columns are looked up by name with
            the first column of that name.
        :param missing_values: Values to consider as missing, or a :class:`MissingValueConfig`
        :param missing_value_repr: Value to use for missing data when outputting
        :param formats: Optional mapping of column names (or indices) to Python format
            strings used when rendering the table. Either a format spec (e.g.
            ``'.2f'``, ``'>8,d'``) or a full format string (e.g. ``'{:.1f} keV'``)
        """
        self._headers: list[str] = list(headers) if headers is not None else []
        self._data: list[list[Any]] = []

        # Handle missing values
        self.default_missing_value: Any = None
        self._missing: MissingValueConfig = self._missing_config(missing_values, missing_value_repr)

        # Handle column formats
        self._formats: dict[Union[str, int], str] = {}
        for col, fmt in (formats or {}).items():
            self.set_format(col, fmt)

        # Process initial data if provided
        if data is not None:
            for row in data:
                self.add_row(row)

    @staticmethod
    def _missing_config(
        values: Union[MissingValueConfig, Any], value_repr: Any = None, checker: Optional[Callable[[Any], bool]] = None
    ) -> MissingValueConfig:
        """A :class:`MissingValueConfig` as given, or made from the values."""
        if isinstance(values, MissingValueConfig):
            return values
        return MissingValueConfig(values=values, repr=value_repr, checker=checker)

    def _process_value(self, value: Any) -> Any:
        """Mark a value as missing if it is one."""
        if value is _EMPTY:
            return MissingValue(self.default_missing_value)
        if isinstance(value, MissingValue):
            return value
        if value in self._missing:
            return MissingValue(value=value)
        return value

    def _to_repr(self, value: Any) -> Any:
        """Convert a value for output."""
        if isinstance(value, MissingValue):
            return self._missing.repr
        return value

    @property
    def missing(self) -> MissingValueConfig:
        """Get the MissingValueConfig instance for this table.

        :return: The MissingValueConfig instance
        """
        return self._missing

    @missing.setter
    def missing(self, value: Union[MissingValueConfig, Any]):
        """Set the missing value configuration for this table. Values without a
        :class:`MissingValueConfig` keep the current representation.

        :param value: MissingValueConfig instance or value(s) to consider missing
        """
        self._missing = self._missing_config(value, self._missing.repr)

        # Check the original values against the new configuration
        self._data = [
            [self._process_value(v.value if isinstance(v, MissingValue) else v) for v in row] for row in self._data
        ]

    ###########
    # Indices #
    ###########
    def _col_index(self, col: Union[str, int]) -> int:
        """Resolve a column name or (negative) index to an index.

        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        :raises TypeError: If the column is neither a name nor an index
        """
        if isinstance(col, str):
            try:
                return self._headers.index(col)
            except ValueError:
                raise KeyError(f'Column {col!r} does not exist') from None
        idx = self._index(col, 'Column')
        n = self.no_cols
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f'Column index {col} is out of range')
        return idx

    def _row_index(self, row: int) -> int:
        """Resolve a (negative) row index to an index.

        :raises IndexError: If the row index is out of range
        :raises TypeError: If the row is not an index
        """
        idx = self._index(row, 'Row')
        n = self.no_rows
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f'Row index {row} is out of range')
        return idx

    @staticmethod
    def _index(value: Any, axis: str) -> int:
        """An integer index, including numpy's integers."""
        if isinstance(value, bool):
            raise TypeError(f'{axis} index must be an integer, not bool')
        try:
            return operator.index(value)
        except TypeError:
            names = 'a name or an index' if axis == 'Column' else 'an integer or a slice'
            raise TypeError(f'{axis} must be {names}, not {type(value).__name__}') from None

    def _col_indices(self, col_slice: slice) -> list[int]:
        """Resolve a column slice to a list of column indices. Column names may be used
        for both ends, and the stop column is then included."""
        start, stop = col_slice.start, col_slice.stop
        if isinstance(start, str):
            start = self._col_index(start)
        if isinstance(stop, str):
            stop = self._col_index(stop) + 1
        return self._slice_indices(slice(start, stop, col_slice.step), self.no_cols, 'Column', col_slice)

    def _row_indices(self, row_slice: slice) -> list[int]:
        """Resolve a row slice to a list of row indices."""
        return self._slice_indices(row_slice, self.no_rows, 'Row', row_slice)

    @classmethod
    def _slice_indices(cls, s: slice, n: int, axis: str, given: slice) -> list[int]:
        """Indices of a slice over ``n`` items: unlike lists, a slice beyond the ends
        raises instead of being clipped.

        :raises IndexError: If the slice is out of bounds
        """
        step = 1 if s.step is None else cls._index(s.step, axis)
        if step == 0:
            raise ValueError('Slice step cannot be zero')
        start = 0 if s.start is None else cls._index(s.start, axis)
        stop = n if s.stop is None else cls._index(s.stop, axis)
        if start < 0:
            start += n
        if stop < 0:
            stop += n
        indices = list(range(start, stop, step))
        if indices and (min(indices) < 0 or max(indices) >= n):
            raise IndexError(f'{axis} slice {given} out of bounds')
        return indices

    def _select(self, col_indices: Iterable[int], row_indices: Iterable[int]) -> 'Table':
        """A new table with the given columns and rows of this one (copied), its missing
        value configuration and the formats of the columns. Without columns, it has no
        rows either."""
        col_indices = list(col_indices)
        row_indices = list(row_indices) if col_indices else []
        headers = [self._headers[i] for i in col_indices] if self._headers else None
        return Table(
            data=[[self._data[r][c] for c in col_indices] for r in row_indices],
            headers=headers,
            missing_values=self._missing,
            formats=self._subset_formats(col_indices, headers),
        )

    ###########
    # Formats #
    ###########
    def _format_key(self, col: Union[str, int]) -> Union[str, int]:
        """Resolve the key under which the format of a column is stored. Tables with
        headers store formats by column name, tables without headers by column index."""
        if isinstance(col, str) or self._headers:
            return self._headers[self._col_index(col)]
        if self.no_cols:
            return self._col_index(col)
        # Formats can be given before the data: any column
        idx = self._index(col, 'Column')
        if idx < 0:
            raise IndexError(f'Column index {col} is out of range')
        return idx

    @property
    def formats(self) -> dict[Union[str, int], str]:
        """Get the column formats used when rendering the table.

        :return: A copy of the column formats
        """
        return dict(self._formats)

    def set_format(self, col: Union[str, int], fmt: Optional[str]):
        """Set the format of a column used when rendering the table.

        :param col: Column name or index
        :param fmt: Python format spec (e.g. ``'.2f'``) or full format string (e.g.
            ``'{:.1f} keV'``). Use ``None`` to clear the format.
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        :raises TypeError: If the format is not a string
        """
        key = self._format_key(col)
        if fmt is None:
            self._formats.pop(key, None)
            return
        if not isinstance(fmt, str):
            raise TypeError(f'Format must be a string, not {type(fmt).__name__}')
        self._formats[key] = fmt

    def get_format(self, col: Union[str, int]) -> Optional[str]:
        """Get the format of a column.

        :param col: Column name or index
        :return: The format string or None if the column has no format
        """
        return self._formats.get(self._format_key(col))

    def _get_format(self, col_idx: int) -> Optional[str]:
        """Get the format of a column by index, without validation."""
        if self._headers:
            if col_idx >= len(self._headers):
                return None
            return self._formats.get(self._headers[col_idx])
        return self._formats.get(col_idx)

    def _subset_formats(
        self, col_indices: Iterable[int], headers: Optional[list[str]] = None
    ) -> dict[Union[str, int], str]:
        """Get the formats of a subset of columns, keyed for a new table.

        :param col_indices: Indices of the columns in this table
        :param headers: Headers of the new table (defaults to the headers of the selected
            columns)
        :return: Formats keyed by the new column names, or by the new column indices if
            there are no headers
        """
        col_indices = list(col_indices)
        if headers is None and self._headers:
            headers = [self._headers[i] for i in col_indices]
        formats = {}
        for j, i in enumerate(col_indices):
            fmt = self._get_format(i)
            if fmt is not None:
                formats[headers[j] if headers else j] = fmt
        return formats

    def _format_cell(self, col_idx: int, value: Any) -> str:
        """Convert a value to a string for output, applying the column format. Missing
        values are never formatted. If the format cannot be applied to the value, then
        ``str(value)`` is returned instead."""
        if isinstance(value, MissingValue):
            return str(self._missing.repr)
        fmt = self._get_format(col_idx)
        if fmt is None:
            return str(value)
        try:
            if '{' in fmt:
                return fmt.format(value)
            return format(value, fmt)
        except (ValueError, TypeError, IndexError, KeyError):
            return str(value)

    @staticmethod
    def _spec_align(fmt: str) -> Optional[str]:
        """Extract the alignment character from a format spec or the first replacement
        field of a format string."""
        if '{' in fmt:
            match = re.search(r'\{[^{}:]*:([^{}]*)\}', fmt)
            if not match:
                return None
            fmt = match.group(1)
        if len(fmt) >= 2 and fmt[1] in '<>^=':
            return fmt[1]
        if fmt and fmt[0] in '<>^=':
            return fmt[0]
        return None

    def _col_align(self, col_idx: int) -> str:
        """Get the alignment of a column (``'<'``, ``'>'`` or ``'^'``). Unformatted
        columns are left-aligned. Formatted columns use the alignment of the format, or
        default to right-aligned if the column contains numbers and left-aligned
        otherwise."""
        fmt = self._get_format(col_idx)
        if fmt is None:
            return '<'
        align = self._spec_align(fmt)
        if align is not None:
            return '>' if align == '=' else align
        if any(isinstance(row[col_idx], Number) and not isinstance(row[col_idx], bool) for row in self._data):
            return '>'
        return '<'

    @staticmethod
    def _pad(s: str, width: int, align: str) -> str:
        """Pad a string to the given width with the given alignment."""
        if align == '>':
            return s.rjust(width)
        if align == '^':
            return s.center(width)
        return s.ljust(width)

    #########
    # Shape #
    #########
    @property
    def headers(self) -> list[str]:
        """Get the column headers. Columns are looked up by name with the first column of
        that name.

        :return: A copy of the column headers
        """
        return list(self._headers)

    @headers.setter
    def headers(self, headers: list[str]):
        """Set the column headers for the table.

        :param headers: List of column names to set as headers
        :raises ValueError: If the number of columns does not match the width of existing rows
        """
        if len(self._data) > 0 and len(headers) != len(self._data[0]):
            raise ValueError('Number of headers does not match row width')
        # Re-key the column formats by position
        n = min(len(headers), max(len(self._headers), self.no_cols))
        formats = [self._get_format(i) for i in range(n)]
        self._headers = list(headers)
        self._formats = {}
        for i, fmt in enumerate(formats):
            if fmt is not None:
                self._formats[self._headers[i] if self._headers else i] = fmt

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
    def shape(self) -> tuple[int, int]:
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
        return any(isinstance(v, MissingValue) for row in self._data for v in row)

    #################
    # Rows, columns #
    #################
    def add_row(self, data: Union[Iterable[Any], Mapping[Union[str, int], Any]]):
        """Append a new row to the table.

        :param data: The row data to append: a list of values in column order, or a
            dictionary mapping column names/indices to values (the other cells are missing
            values)
        :raises ValueError: If the row length does not match the number of columns
        :raises KeyError: If a dictionary key is a column name that doesn't exist
        :raises IndexError: If a dictionary key is a column index that is out of range
        """
        if isinstance(data, Mapping):
            row = [_EMPTY] * self.no_cols
            for key, value in data.items():
                row[self._col_index(key)] = value
        else:
            row = list(data)
            if (self._headers or self._data) and len(row) != self.no_cols:
                raise ValueError(f'Row length {len(row)} does not match the number of columns {self.no_cols}')
        self._data.append([self._process_value(v) for v in row])

    def add_col(
        self,
        data: Union[Iterable[Any], Mapping[str, Iterable[Any]], str, None] = None,
        *,
        col_name: Optional[str] = None,
        fmt: Optional[str] = None,
    ):
        """Append a new column to the table. On a table without rows, the column's values
        make the rows, and the other columns get missing values.

        :param data: The column data to append, a dictionary of the column name to its
            data, or the name of a column of missing values
        :param col_name: The name of the new column
        :param fmt: Optional format of the new column (see :meth:`set_format`)
        :raises ValueError: If the number of values does not match the number of rows
        :raises ValueError: If the table has headers but no column name is provided, or
            the other way round
        """
        if isinstance(data, str):
            # Handle creation of empty columns
            col_name, data = data, None
        elif isinstance(data, Mapping):
            if len(data) != 1:
                raise ValueError('Only one column can be added at a time with a dict input')
            ((name, data),) = data.items()
            if col_name is None:
                col_name = name

        old_cols = self.no_cols
        if col_name is None and self._headers:
            raise ValueError('Column name must be provided when table has headers')
        if col_name is not None and not self._headers and old_cols:
            raise ValueError('Column name given, but the other columns have no headers')

        # If no data is provided, create a column of missing values
        values = [_EMPTY] * self.no_rows if data is None else list(data)
        if self._data and len(values) != self.no_rows:
            raise ValueError(f'Column length {len(values)} does not match the number of rows {self.no_rows}')

        if col_name is not None:
            self._headers.append(col_name)
        if not self._data:
            self._data = [[self._process_value(_EMPTY) for _ in range(old_cols)] for _ in values]
        for row, value in zip(self._data, values):
            row.append(self._process_value(value))

        if fmt is not None:
            self.set_format(self.no_cols - 1, fmt)

    def get_row(self, idx: int) -> list[Any]:
        """Get a row by index.

        :param idx: The row index
        :raises IndexError: If the row index is out of range
        """
        return [self._to_repr(v) for v in self._data[self._row_index(idx)]]

    def get_col(self, idx: Union[str, int]) -> list[Any]:
        """Get a column by name or index.

        :param idx: The column name or index
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        """
        col = self._col_index(idx)
        return [self._to_repr(row[col]) for row in self._data]

    def set_row(self, idx: int, row: Iterable[Any]):
        """Set the data for a row at a given index.

        :param idx: The row index
        :param row: The new row data
        :raises IndexError: If the row index is out of range
        :raises ValueError: If the row length does not match the number of columns
        """
        r = self._row_index(idx)
        row = list(row)
        if len(row) != self.no_cols:
            raise ValueError(f'Row length {len(row)} does not match the number of columns {self.no_cols}')
        self._data[r] = [self._process_value(v) for v in row]

    def set_col(self, idx: Union[str, int], values: Iterable[Any]):
        """Set the data for a column by name or index.

        :param idx: The column name or index
        :param values: The new column data
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        :raises ValueError: If the number of values does not match the number of rows
        """
        col = self._col_index(idx)
        values = list(values)
        if len(values) != self.no_rows:
            raise ValueError(f'Column length {len(values)} does not match the number of rows {self.no_rows}')
        for row, value in zip(self._data, values):
            row[col] = self._process_value(value)

    def del_row(self, idx: int):
        """Remove a row at the given index.

        :param idx: The row index to remove
        :raises IndexError: If the row index is out of range
        """
        self._data.pop(self._row_index(idx))

    def del_col(self, idx: Union[str, int]):
        """Remove a column by name or index.

        :param idx: The column name or index to remove
        :raises KeyError: If the column name does not exist
        :raises IndexError: If the column index is out of range
        """
        col = self._col_index(idx)
        for row in self._data:
            row.pop(col)

        if self._headers:
            col_name = self._headers.pop(col)
            if col_name not in self._headers:
                self._formats.pop(col_name, None)
        else:
            # Shift index-keyed formats
            self._formats = {
                (k - 1 if k > col else k): v for k, v in self._formats.items() if isinstance(k, int) and k != col
            }

    ###############
    # Conversions #
    ###############
    @property
    def data(self) -> list[list[Any]]:
        """Get the raw data of the table.

        :return: The table data as a list of rows
        """
        return [[self._to_repr(v) for v in row] for row in self._data]

    def to_list(self) -> list[Any]:
        """Convert the table to a flattened list

        :return: The table data as a list
        """
        return [self._to_repr(v) for row in self._data for v in row]

    def to_dict(self) -> dict[Union[str, int], list[Any]]:
        """Convert the table to a dictionary with column names as keys, or column indices
        for tables without headers. Of columns with the same name, the first is kept.

        :return: The table data as a dictionary
        """
        if self._headers:
            return {col: self.get_col(col) for col in self._headers}
        return {i: self.get_col(i) for i in range(self.no_cols)}

    def to_numpy(self) -> 'np.ndarray':
        """Convert the table to a numpy ndarray. Requires numpy to be installed.

        :return: The table as a numpy array
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise ImportError(f'numpy is not installed; {_EXTRA_HINT}')

        return np.array(self.data)

    def to_pandas(self) -> 'pd.DataFrame':
        """Convert the table to a pandas DataFrame. Requires pandas to be installed.

        :return: The table as a DataFrame
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ImportError(f'pandas is not installed; {_EXTRA_HINT}')

        # An `Index`, which pandas 2's annotations accept, unlike a list of names
        return pd.DataFrame(self.data, columns=pd.Index(self._headers) if self._headers else None)

    def to_rich(self, cast_as: Optional[Callable] = None) -> 'rich.table.Table':
        """Convert the table to a rich.Table for pretty console output. Requires rich to
        be installed.

        :param cast_as: Optional function to cast values before adding to the rich table
        :return: The table as a rich Table object
        """
        try:
            from rich.table import Table as RichTable
        except ModuleNotFoundError:
            raise ImportError(f'rich is not installed; {_EXTRA_HINT}')

        justify: dict[str, Literal['left', 'right', 'center']] = {'<': 'left', '>': 'right', '^': 'center'}
        table = RichTable()
        if self._headers:
            for i, col in enumerate(self._headers):
                table.add_column(col, justify=justify[self._col_align(i)])

        for row in self._data:
            if cast_as:
                values = [cast_as(self._to_repr(v)) for v in row]
            else:
                values = [None if self._to_repr(v) is None else self._format_cell(i, v) for i, v in enumerate(row)]
            table.add_row(*values)
        return table

    def to_csv(
        self,
        filename: Optional[str | Path] = None,
        delimiter: str = ',',
        new_line: str = '\n',
        header_char: str = '',
        overwrite: bool = False,
        keep_file_ext: bool = False,
        formatted: bool = True,
        encoding: str = 'utf-8',
    ) -> str | Path:
        """Write the table to a CSV file. If ``filename`` is not provided, then the
        CSV will be returned as a string. Values with the delimiter, quotes or line breaks
        are quoted, as in :mod:`csv`.

        :param filename: Optional output path for the CSV file.
        :param delimiter: Delimiter to use in the CSV, one character.
        :param new_line: Newline character to use in the CSV.
        :param header_char: Character to prepend to the header line (e.g., '#').
        :param overwrite: Overwrite the file if it already exists.
        :param keep_file_ext: Keep the file extension if ``filename`` is provided.
        :param formatted: Apply the column formats to the values (without padding).
        :param encoding: Encoding of the file.
        :return: Either the CSV string or the output path.
        :raises FileExistsError: If the file already exists and ``overwrite`` is False
        """

        def cell(i: int, value: Any) -> str:
            if not formatted:
                return str(self._to_repr(value))
            text = self._format_cell(i, value)
            # Only the padding of a format goes, not the value's own spaces
            if isinstance(value, MissingValue) or self._get_format(i) is None:
                return text
            return text.strip()

        def line(values: list[str]) -> str:
            buffer = io.StringIO()
            csv.writer(buffer, delimiter=delimiter, lineterminator=new_line).writerow(values)
            return buffer.getvalue()

        lines = [header_char + line(self._headers)] if self._headers else []
        lines += [line([cell(i, v) for i, v in enumerate(row)]) for row in self._data]
        result = ''.join(lines)

        # If no filename is provided, then return a CSV string
        if filename is None:
            return result

        # If a filename is provided, then write to a file
        filename = Path(filename)
        if filename.suffix != '.csv' and not keep_file_ext:
            filename = filename.with_suffix('.csv')
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File already exists: {filename}')
        filename.parent.mkdir(parents=True, exist_ok=True)

        # The line breaks as given, not the system's
        filename.write_text(result, encoding=encoding, newline='')
        return filename

    ################
    # Constructors #
    ################
    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Iterable[Any]],
        *,
        missing_values: Union[MissingValueConfig, Any] = (),
        missing_value_repr: Any = None,
        formats: Optional[dict[Union[str, int], str]] = None,
    ) -> 'Table':
        """Create a Table from a dictionary of columns.

        :param data: Dictionary with column names as keys and column data as values
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        :param formats: Column formats used when rendering the table
        :return: A new Table instance
        :raises ValueError: If the columns are not of equal length
        """
        columns = [list(col) for col in data.values()]
        if len({len(col) for col in columns}) > 1:
            raise ValueError('All columns must have the same length')
        return cls(
            data=list(zip(*columns)),
            headers=list(data.keys()),
            missing_values=missing_values,
            missing_value_repr=missing_value_repr,
            formats=formats,
        )

    @classmethod
    def from_numpy(
        cls,
        array: 'np.ndarray',
        *,
        headers: Optional[list[str]] = None,
        missing_values: Union[MissingValueConfig, Any] = (),
        missing_value_repr: Any = None,
        formats: Optional[dict[Union[str, int], str]] = None,
    ) -> 'Table':
        """Create a Table from a numpy ndarray. NaN values are missing values, unless a
        :class:`MissingValueConfig` is given.

        :param array: 2D numpy array
        :param headers: Optional list of column names
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        :param formats: Column formats used when rendering the table
        :return: A new Table instance
        :raises ImportError: If numpy is not installed
        :raises ValueError: If the array is not 2D
        """
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            raise ImportError(f'numpy is not installed; {_EXTRA_HINT}')

        if len(array.shape) != 2:
            raise ValueError('Array must be 2D')

        return cls(
            data=array.tolist(),
            headers=headers,
            missing_values=cls._missing_config(missing_values, missing_value_repr, checker=_is_nan),
            formats=formats,
        )

    @classmethod
    def from_pandas(
        cls,
        df: 'pd.DataFrame',
        *,
        missing_values: Union[MissingValueConfig, Any] = (),
        missing_value_repr: Any = None,
        formats: Optional[dict[Union[str, int], str]] = None,
    ) -> 'Table':
        """Create a Table from a pandas DataFrame. Values that pandas considers missing
        (``None``, NaN, ``NaT``, ...) are missing values, unless a
        :class:`MissingValueConfig` is given.

        :param df: pandas DataFrame
        :param missing_values: Values to consider as missing
        :param missing_value_repr: Value to use for missing data when outputting
        :param formats: Column formats used when rendering the table
        :return: A new Table instance
        :raises ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ImportError(f'pandas is not installed; {_EXTRA_HINT}')

        def is_na(value: Any) -> bool:
            # `pd.isna()` of a list or an array is element-wise
            return pd.api.types.is_scalar(value) and bool(pd.isna(value))

        return cls(
            data=df.values.tolist(),
            headers=df.columns.tolist(),
            missing_values=cls._missing_config(missing_values, missing_value_repr, checker=is_na),
            formats=formats,
        )

    @classmethod
    def from_csv(
        cls,
        s: str | Path,
        *,
        delimiter: str = ',',
        header_line: Optional[int] = None,
        header_char: str = '',
        formats: Optional[dict[Union[str, int], str]] = None,
        encoding: str = 'utf-8',
    ) -> 'Table':
        """Create a Table from a CSV file or string. Quoted values are read as in
        :mod:`csv`; all values are strings.

        A :class:`~pathlib.Path` is a file. A string is the CSV content if it has a line
        break, otherwise a file if one of that name exists, else the CSV content.

        :param s: Path to CSV file or CSV string content
        :param delimiter: Delimiter used in the CSV, one character (default: ',')
        :param header_line: Line that contains headers (default: None); the lines before
            it are skipped
        :param header_char: Character that might prefix the header line (e.g., '#')
        :param formats: Column formats used when rendering the table
        :param encoding: Encoding of the file.
        :return: A new Table instance
        """
        # Check if the input is empty
        if not s:
            return cls(headers=[], formats=formats)

        if isinstance(s, os.PathLike) or ('\n' not in s and '\r' not in s and path_is_file(s)):
            # Line breaks as they are, for quoted values that span lines
            with open(s, encoding=encoding, newline='') as f:
                content = f.read()
        else:
            content = s

        # Split at line breaks only, keeping them for the reader
        lines = list(io.StringIO(content, newline=''))

        headers = None
        if header_line is not None:
            lines = lines[header_line:]
            # Remove header character if present, before the quotes are parsed
            if header_char and lines and lines[0].startswith(header_char):
                lines[0] = lines[0][len(header_char) :]

        reader = csv.reader(lines, delimiter=delimiter)
        if header_line is not None:
            headers = next(reader, [])
        # Skip empty lines
        records = [record for record in reader if record]

        return cls(data=records, headers=headers, formats=formats)

    #############
    # Operators #
    #############
    def __str__(self):
        """Return a pretty-printed string representation of the table.

        :return: String representation of the table
        """
        if not self._data:
            return '(Empty table)'

        # Convert values to strings, handling missing values and column formats
        data = [[self._format_cell(i, v) for i, v in enumerate(row)] for row in self._data]

        if not self._headers:
            # Print as plain rows
            return '\n'.join(' | '.join(row) for row in data)

        col_widths = [
            max(len(str(col)), max((len(row[i]) for row in data), default=0)) for i, col in enumerate(self._headers)
        ]
        col_aligns = [self._col_align(i) for i in range(len(self._headers))]
        header = ' | '.join(self._pad(str(col), col_widths[i], col_aligns[i]) for i, col in enumerate(self._headers))
        sep = '-+-'.join('-' * w for w in col_widths)
        rows = [
            ' | '.join(self._pad(row[i], col_widths[i], col_aligns[i]) for i in range(len(self._headers)))
            for row in data
        ]
        return '\n'.join([header, sep] + rows)

    def __len__(self):
        """Return the number of rows in the table."""
        return self.no_rows

    def __getitem__(self, key):
        """Get data from the table using pandas-like slicing syntax.

        Supports:
        - table[col] -> New Table with a single column (by name or index)
        - table[col0:col1] -> New Table with selected columns
        - table[col, row] -> Single cell value
        - table[col, row0:row1] -> New Table with selected part of a column
        - table[col0:col1, row] -> New Table with a single row
        - table[col0:col1, row0:row1] -> New Table with subset of the table

        :param key: Index, slice, string, or tuple for advanced indexing
        :return: Single value, or Table instance (multiple values)
        :raises IndexError: If indices are out of range
        :raises KeyError: If column names don't exist
        :raises TypeError: If the key is neither of the above
        """
        if isinstance(key, tuple):
            col_spec, row_spec = key
            if isinstance(col_spec, slice):
                cols = self._col_indices(col_spec)
            else:
                cols = [self._col_index(col_spec)]

            if isinstance(row_spec, slice):
                return self._select(cols, self._row_indices(row_spec))
            row = self._row_index(row_spec)
            if not isinstance(col_spec, slice):
                return self._to_repr(self._data[row][cols[0]])
            return self._select(cols, [row])

        if isinstance(key, slice):
            return self._select(self._col_indices(key), range(self.no_rows))
        return self._select([self._col_index(key)], range(self.no_rows))

    def __setitem__(self, key, value):
        """Set data in the table using pandas-like slicing syntax.

        Supports:
        - table[col] = values -> Set an entire column
        - table[col, row] = value -> Set a single cell
        - table[col, row0:row1] = values -> Set part of a column; a single value (a
          string too) is set in every cell

        :param key: Index, slice, string, or tuple for advanced indexing
        :param value: The value(s) to set
        :raises IndexError: If indices are out of range
        :raises KeyError: If column names don't exist
        :raises ValueError: If value dimensions don't match the target
        :raises NotImplementedError: If several columns are selected
        """
        if isinstance(key, slice) or (isinstance(key, tuple) and isinstance(key[0], slice)):
            raise NotImplementedError('Setting multiple columns with a slice is not supported')

        if not isinstance(key, tuple):
            self.set_col(key, value)
            return

        col_spec, row_spec = key
        col = self._col_index(col_spec)
        if isinstance(row_spec, slice):
            rows = self._row_indices(row_spec)
            if isinstance(value, (str, bytes)) or not hasattr(value, '__len__'):
                values = [value] * len(rows)
            else:
                values = list(value)
                if len(values) != len(rows):
                    raise ValueError(f'Value length {len(values)} does not match target length {len(rows)}')
        else:
            rows, values = [self._row_index(row_spec)], [value]

        for r, v in zip(rows, values):
            self._data[r][col] = self._process_value(v)

    def __iter__(self) -> Iterator[list[Any]]:
        """Iterate over rows in the table."""
        for row in self._data:
            yield [self._to_repr(val) for val in row]

    def items(self) -> Iterator[tuple[str, list[Any]]]:
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
        if self._headers != other._headers:
            raise ValueError('Tables must have the same column headers to be concatenated')

        # Use the same missing value configuration
        return Table(
            data=self._data + other._data,
            headers=self._headers,
            missing_values=self._missing,
            formats=self._subset_formats(range(self.no_cols)),
        )

    def __or__(self, other: 'Table') -> 'Table':
        """Concatenate columns from another table to this table.

        The two tables must have the same number of rows.

        :param other: The table to append columns from
        :return: A new table with combined columns
        :raises ValueError: If the tables have different numbers of rows
        """
        if not isinstance(other, Table):
            raise TypeError(f"Cannot use '|' operator between Table and {type(other).__name__}")
        if len(self) != len(other):
            raise ValueError('Tables must have the same number of rows to concatenate columns')

        # Combine formats, preferring the formats of the first table
        formats = other._subset_formats(range(other.no_cols))
        if not other._headers:
            formats = {(k + self.no_cols if isinstance(k, int) else k): v for k, v in formats.items()}
        formats.update(self._subset_formats(range(self.no_cols)))

        # For missing values, prefer the configuration from the first table
        return Table(
            data=[a + b for a, b in zip(self._data, other._data)],
            headers=self._headers + other._headers,
            missing_values=self._missing,
            formats=formats,
        )

    def __iadd__(self, other: Union['Table', Iterable[Any], Mapping[Union[str, int], Any]]) -> 'Table':
        """In-place row concatenation (append rows from another table or add a new row).

        The method can be used in two ways:
        1. Append rows from another table (tables must have the same column structure)
        2. Add a single row as a list or dictionary

        :param other: The table to append rows from or the row data to append
        :return: Self with rows from other table appended or with new row added
        :raises ValueError: If the tables have different column headers
        """
        if isinstance(other, Table):
            if self._headers != other._headers:
                raise ValueError('Tables must have the same column headers to be concatenated')
            # Copies of the rows, which also keep `self += self` finite
            for row in [list(row) for row in other._data]:
                self.add_row(row)
        else:
            self.add_row(other)
        return self

    def __ior__(self, other: Union['Table', Iterable[Any], Mapping[str, Iterable[Any]], str]) -> 'Table':
        """In-place column concatenation (append columns from another table or add a new column).

        The method can be used in two ways:
        1. Append columns from another table (tables must have the same number of rows)
        2. Add a single column as a list, dictionary, or string (column name with default missing values)

        :param other: The table to append columns from, column data to append, or column name
        :return: Self with columns from other table appended or with new column added
        :raises ValueError: If the tables have different numbers of rows
        """
        if not isinstance(other, Table):
            self.add_col(other)
            return self
        if len(self) != len(other):
            raise ValueError('Tables must have the same number of rows to concatenate columns')

        # Copies first, which also keep `self |= self` right
        headers = list(other._headers)
        rows = [list(row) for row in other._data]
        formats = other._subset_formats(range(other.no_cols))

        # Append the formats from other table, keeping existing ones
        no_cols = self.no_cols
        for key, fmt in formats.items():
            # Index keys (the other table has no headers) move past this table's columns
            self._formats.setdefault(key + no_cols if isinstance(key, int) and not headers else key, fmt)

        self._headers.extend(headers)
        for row, extra in zip(self._data, rows):
            row.extend(self._process_value(v) for v in extra)
        return self

    def __sub__(self, other: Union['Table', str, list[str], tuple[str, ...]]) -> 'Table':
        """Remove columns from the table or compute the column-wise difference with another table.

        This method can be used in two ways:
        1. table - other_table: Return a new table with only the columns that are in this table
           but not in the other table (column-wise difference)
        2. table - 'col_name': Return a new table with the specified column removed
        3. table - ['col1', 'col2']: Return a new table with multiple columns removed

        A table without columns has no rows either.

        :param other: The table to compute difference with, or column(s) to remove
        :return: A new table with columns removed
        :raises KeyError: If a column name does not exist
        """
        if isinstance(other, Table):
            remove = other._headers
        elif isinstance(other, (str, list, tuple)):
            remove = [other] if isinstance(other, str) else list(other)
            for col in remove:
                if col not in self._headers:
                    raise KeyError(f'Column {col!r} does not exist')
        else:
            raise TypeError(f'Cannot subtract {type(other).__name__} from Table')

        keep = [i for i, col in enumerate(self._headers) if col not in remove]
        return self._select(keep, range(self.no_rows))

    def __isub__(self, other: Union['Table', str, list[str], tuple[str, ...]]) -> 'Table':
        """In-place removal of columns from the table. The rows stay, also when no column does.

        This method can be used in two ways:
        1. table -= other_table: Remove columns from this table that are also in the other table
        2. table -= 'col_name': Remove the specified column from this table
        3. table -= ['col1', 'col2']: Remove multiple columns from this table

        :param other: The table to compute difference with, or column(s) to remove
        :return: Self with columns removed
        :raises KeyError: If a column name does not exist
        """
        if isinstance(other, Table):
            remove = [col for col in self._headers if col in other._headers]
        elif isinstance(other, (str, list, tuple)):
            remove = [other] if isinstance(other, str) else list(other)
        else:
            raise TypeError(f'Cannot subtract {type(other).__name__} from Table')

        # All are checked before any is removed; in reverse order to avoid index shifting
        for col in sorted({self._col_index(col) for col in remove}, reverse=True):
            self.del_col(col)
        return self
