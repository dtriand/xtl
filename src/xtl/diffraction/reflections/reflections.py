import copy
from pathlib import Path
from typing import Any, Iterable
from typing_extensions import Self

import gemmi
import reciprocalspaceship as rs
from reciprocalspaceship.dtypes.base import MTZDtype
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas._typing import Dtype, ArrayLike, NpDtype

from xtl.diffraction.reflections.files import MTZ_COLUMN_TYPES, MTZ_DTYPES
from xtl.diffraction.reflections.metadata import *


class Reflection:
    # Representation of a single reflection with one or more columns of data
    ...


class _ReflectionsBase:

    def __init__(self,
                 # pd.DataFrame arguments
                 data: ArrayLike | Iterable | dict | pd.DataFrame | rs.DataSet |
                       gemmi.Mtz,
                 index: ArrayLike | pd.Index = None,
                 columns: ArrayLike | pd.Index = None,
                 types: Iterable | pd.Index | pd.Series | Dtype | str |
                        dict[str, Dtype | str] = None,
                 copy: bool = False,
                 # Additional rs.DataSet arguments
                 space_group: gemmi.SpaceGroup = None,
                 unit_cell: gemmi.UnitCell = None,
                 merged: bool = None,
                 # XTL arguments
                 metadata: ReflectionsMetadataType = None
                 ):
        # Extract space group and unit-cell from gemmi.Mtz if not provided
        if isinstance(data, gemmi.Mtz):
            space_group = space_group or data.spacegroup
            unit_cell = unit_cell or data.cell
            metadata = metadata or MTZReflectionsMetadata.from_gemmi(data)
        self._rs = rs.DataSet(data=data, index=index, columns=columns, dtype=None,
                              copy=copy, spacegroup=space_group, cell=unit_cell,
                              merged=merged)

        # Store metadata
        if metadata:
            if not isinstance(metadata, ReflectionsMetadata):
                raise TypeError(f'Expected {ReflectionsMetadata.__class__.__name__}, '
                                f'got {type(metadata)}')
            self._metadata = metadata
        else:
            self._metadata = None

        # Set dtypes
        if types is not None:
            if hasattr(self, 'as_types'):  # for ReflectionsCollection
                self.as_types(types)
            elif hasattr(self, 'as_type'):  # for ReflectionsData
                self.as_type(types)

    def _get_dtype(self, t: Any) -> Any:
        """
        Convert a type to a MTZDtype.
        """
        if isinstance(t, MTZDtype):
            return t
        elif isinstance(t, str):
            if t not in MTZ_COLUMN_TYPES:
                raise ValueError(f'Invalid MTZ type {t!r}. '
                                 f'Choose one from: {", ".join(MTZ_COLUMN_TYPES)}')
            return MTZ_DTYPES[t]
        elif isinstance(t, dict):
            return {k: self._get_dtype(v) for k, v in t.items()}
        elif isinstance(t, Iterable | pd.Index):
            return [self._get_dtype(st) for st in t]
        elif isinstance(t, ExtensionDtype | NpDtype):  # proxy for Dtype
            return t
        raise TypeError(f'Expected MTZDtype, str or Dtype, got {type(t)}')

    @property
    def metadata(self) -> ReflectionsMetadataType | None:
        """
        Metadata associated with the reflections when instantiated from a file.
        """
        return self._metadata

    def dropna(self, inplace: bool = False) -> Self:
        """
        Drop reflections with missing data.
        """
        if inplace:
            self._rs.dropna(inplace=True)
            return self
        else:
            r = copy.deepcopy(self)
            r._rs.dropna(inplace=True)
            return r

    def drop_missing(self, inplace: bool = False) -> Self:
        """
        Drop reflections with missing data.
        """
        return self.dropna(inplace=inplace)

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the reflections array.
        """
        return self._rs.shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the reflections array.
        """
        return self._rs.ndim

    @property
    def no_reflections(self) -> int:
        """
        The number of reflections in the array.
        """
        return self.shape[0]

    @property
    def no_columns(self) -> int:
        """
        The number of columns in the array.
        """
        return self.shape[1]

    @property
    def hkls(self) -> NDArray[np.int32]:  # shape: (n, 3)
        """
        An array of the HKL indices with (n, 3) shape.
        """
        return self._rs.get_hkls()

    @property
    def space_group(self) -> gemmi.SpaceGroup:
        """
        The space group associated with the reflections.
        """
        return self._rs.spacegroup

    @space_group.setter
    def space_group(self, space_group: gemmi.SpaceGroup | str | int):
        """
        Set the space group of the reflections.
        """
        self._rs.spacegroup = space_group

    @property
    def unit_cell(self) -> gemmi.UnitCell:
        """
        The unit-cell associated with the reflections.
        """
        return self._rs.cell

    @unit_cell.setter
    def unit_cell(self, unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                                   tuple[float | int]):
        """
        Set the unit-cell of the reflections.
        """
        self._rs.cell = unit_cell

    @property
    def is_merged(self) -> bool:
        """
        Whether the reflections are merged or not.
        """
        return self._rs.merged

    @is_merged.setter
    def is_merged(self, merged: bool):
        """
        Set the merged status of the reflections.
        """
        if not isinstance(merged, bool) and merged is not None:
            raise TypeError(f'Must be a boolean, got {type(merged)}')
        self._rs.merged = merged

    @property
    def resolution_high(self) -> np.float64:
        """
        The upper resolution limit of the reflections, in Angstroms.
        """
        return self.min_d

    @property
    def resolution_low(self) -> np.float64:
        """
        The lower resolution limit of the reflections, in Angstroms.
        """
        return self.max_d

    @property
    def resolution_range(self) -> tuple[np.float64, np.float64]:
        """
        The resolution range of the reflections, in Angstroms, as a (low, high) tuple.

        Ranges are always returned from low to high scattering angle.
        """
        return self.range_d

    @property
    def min_d(self) -> np.float64:
        """
        The minimum d-spacing of the reflections, in Angstroms.
        """
        return np.min(self.get_array_d())

    @property
    def max_d(self) -> np.float64:
        """
        The maximum d-spacing of the reflections, in Angstroms.
        """
        return np.max(self.get_array_d())

    @property
    def range_d(self) -> tuple[np.float64, np.float64]:
        """
        The d-spacing range of the reflections, in Angstroms, as a (low, high) tuple.

        Ranges are always returned from low to high scattering angle.
        """
        d = self.get_array_d()
        return np.max(d), np.min(d)

    def get_array_d(self) -> NDArray[np.float64]:  # shape: (n,)
        """
        Calculate the d-spacing for all reflections.
        """
        return self.unit_cell.calculate_d_array(self.hkls)

    @property
    def min_1_d(self) -> np.float64:
        """
        The minimum 1/d value of the reflections, in 1/Angstroms.
        """
        return np.min(self.get_array_1_d())

    @property
    def max_1_d(self) -> np.float64:
        """
        The maximum 1/d value of the reflections, in 1/Angstroms.
        """
        return np.max(self.get_array_1_d())

    @property
    def range_1_d(self) -> tuple[np.float64, np.float64]:
        """
        The 1/d range of the reflections, in 1/Angstroms, as a (min, max) tuple.

        Ranges are always returned from low to high scattering angle.
        """
        _1_d = self.get_array_1_d()
        return np.min(_1_d), np.max(_1_d)

    def get_array_1_d(self) -> NDArray[np.float64]:  # shape: (n,)
        """
        Calculate the 1/d value for all reflections, in 1/Angstroms.
        """
        return np.sqrt(self.unit_cell.calculate_1_d2_array(self.hkls))

    @property
    def min_1_d2(self) -> np.float64:
        """
        The minimum 1/d^2 value of the reflections, in 1/Angstroms^2.
        """
        return np.min(self.get_array_1_d2())

    @property
    def max_1_d2(self) -> np.float64:
        """
        The maximum 1/d^2 value of the reflections, in 1/Angstroms^2.
        """
        return np.max(self.get_array_1_d2())

    @property
    def range_1_d2(self) -> tuple[np.float64, np.float64]:
        """
        The 1/d^2 range of the reflections, in 1/Angstroms^2, as a (min, max) tuple.

        Ranges are always returned from low to high scattering angle.
        """
        _1_d2 = self.get_array_1_d2()
        return np.min(_1_d2), np.max(_1_d2)

    def get_array_1_d2(self) -> NDArray[np.float64]:  # shape: (n,)
        """
        Calculate the 1/d^2 value for all reflections, in 1/Angstroms^2.
        """
        return self.unit_cell.calculate_1_d2_array(self.hkls)

    @property
    def min_hkl(self) -> NDArray[np.int32]:  # shape: (3,)
        """
        The HKL indices of the reflection with the lowest resolution.
        """
        return self.hkls[np.argmax(self.get_array_d())]

    @property
    def max_hkl(self) -> NDArray[np.int32]:  # shape: (3,)
        """
        The HKL indices of the reflection with the highest resolution.
        """
        return self.hkls[np.argmin(self.get_array_d())]


class ReflectionsData(_ReflectionsBase):

    def __init__(self,
                 # pd.DataFrame arguments
                 data: ArrayLike | Iterable | dict | pd.DataFrame | rs.DataSet |
                       gemmi.Mtz,
                 index: ArrayLike | pd.Index = None,
                 columns: ArrayLike | pd.Index = None,
                 types: Iterable | pd.Index | pd.Series | Dtype | str |
                        dict[str, Dtype | str] = None,
                 copy: bool = False,
                 # Additional rs.DataSet arguments
                 space_group: gemmi.SpaceGroup = None,
                 unit_cell: gemmi.UnitCell = None,
                 merged: bool = None,
                 # XTL arguments
                 metadata: ReflectionsMetadataType = None
                 ):
        super().__init__(data=data, index=index, columns=columns, types=types,
                         copy=copy, space_group=space_group, unit_cell=unit_cell,
                         merged=merged, metadata=metadata)

        # Check shape
        if self._rs.shape[1] != 1:
            raise ValueError(f'{self.__class__.__name__} must have 1 column, but got '
                             f'{self._rs.shape[1]} columns')

    @property
    def dtype(self) -> MTZDtype:
        """
        The data type of the ReflectionsData
        """
        return self._rs.dtypes.iloc[0]

    def as_type(self, dtype: MTZDtype | str, copy: bool = False) -> None:
        """
        Convert the ReflectionsData to a different data type.

        :param dtype: MTZDtype or MTZ column type as string
        :param copy: Whether to create a copy of the data before converting.
        """
        if copy:
            self._rs = self._rs.copy()
        converted_dtype = self._get_dtype(dtype)
        self._rs = self._rs.astype(converted_dtype)

    @property
    def mtz_type(self) -> str:
        """
        The MTZ type of the ReflectionsData
        """
        return self.dtype.mtztype

    @property
    def column(self) -> pd.Index:
        """
        The column name of the ReflectionsData
        """
        return self._rs.columns

    @property
    def label(self) -> str:
        """
        The label name of the ReflectionsData
        """
        return self.column[0]

    @property
    def values(self) -> rs.DataSeries:
        """
        The data of the ReflectionsData
        """
        return self._rs[self.label]

    def __getitem__(self, item):
        # ToDo: More sophisticated implementation needed
        return self._rs[self.label][item]

    def __setitem__(self, key, value) -> None:
        self._rs.loc[key, self.label] = value

    @classmethod
    def from_rs(cls, dataset: rs.DataSet, label: str,
                space_group: gemmi.SpaceGroup | str | int = None,
                unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                           tuple[float | int] = None,
                merged: bool = None):
        """
        Create a ReflectionsData from a reciprocalspaceship DataSet.
        """
        if not isinstance(dataset, rs.DataSet):
            raise TypeError(f'Expected reciprocalspaceship.DataSet, got {type(dataset)}')
        if label not in dataset.columns:
            raise ValueError(f'Label {label!r} not found in DataSet columns. '
                             f'Choose one from: {", ".join(dataset.columns)}')
        # Extract space group, unit-cell and merged from rs.DataSet if not provided
        if space_group is None:
            space_group = dataset.spacegroup
        if unit_cell is None:
            unit_cell = dataset.cell
        if merged is None:
            merged = dataset.merged
        ic = dataset.columns.get_loc(label)
        return cls(data=dataset[label], index=dataset.index,
                   columns=dataset.columns[ic:ic+1], types=dataset.dtypes[label],
                   copy=True, space_group=space_group, unit_cell=unit_cell,
                   merged=merged)

    @classmethod
    def _from_rs_io(cls, file_type: str, file_reader: callable,
                    file: Path | str, label: str,
                    space_group: gemmi.SpaceGroup | str | int = None,
                    unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                               tuple[float | int] = None,
                    merged: bool = None):
        """
        Create a ReflectionsData from a reciprocalspaceship reader function.
        """
        # Check if file exists
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f'{file_type} file {file} not found')
        # Read the file
        dataset: rs.DataSet = file_reader(str(file))
        return cls.from_rs(dataset=dataset, label=label, space_group=space_group,
                           unit_cell=unit_cell, merged=merged)

    @classmethod
    def from_mtz(cls, mtz_file: Path | str, label: str,
                 space_group: gemmi.SpaceGroup | str | int = None,
                 unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                            tuple[float | int] = None,
                 merged: bool = None):
        """
        Create a ReflectionsData from an MTZ file and a column label.
        """
        return cls._from_rs_io(file_type='MTZ', file_reader=rs.read_mtz,
                                file=mtz_file, label=label, space_group=space_group,
                                unit_cell=unit_cell, merged=merged)

    @classmethod
    def from_cif(cls, cif_file: Path | str, label: str,
                 space_group: gemmi.SpaceGroup | str | int = None,
                 unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                            tuple[float | int] = None,
                 merged: bool = None):
        """
        Create a ReflectionsData from an CIF file and a column label.
        """
        return cls._from_rs_io(file_type='CIF', file_reader=rs.read_cif,
                                file=cif_file, label=label, space_group=space_group,
                                unit_cell=unit_cell, merged=merged)



class ReflectionsCollection(_ReflectionsBase):

    @property
    def dtypes(self) -> pd.Series:
        """
        The data types of the ReflectionsCollection
        """
        return self._rs.dtypes

    def as_type(self, dtypes: Iterable | pd.Index | dict[str, Dtype | str],
                copy: bool = False) -> None:
        """
        Convert the ReflectionsCollection to different data types.

        :params dtypes: Data types to convert to
        :param copy: Whether to create a copy of the data before converting.
        """
        # Convert input to appropriate dtypes
        converted_dtypes = self._get_dtype(dtypes)

        # Apply the conversion
        if isinstance(converted_dtypes, dict):
            if len(converted_dtypes.keys()) != self.no_columns:
                raise ValueError(f'Expected {self.no_columns} dtypes, '
                                 f'got {len(converted_dtypes.keys())}')
        elif isinstance(converted_dtypes, Iterable | pd.Index):
            if len(converted_dtypes) != self.no_columns:
                raise ValueError(f'Expected {self.no_columns} dtypes, got {len(converted_dtypes)}')
            converted_dtypes = {col: dt for col, dt in zip(self.columns, converted_dtypes)}
        else:
            raise TypeError(f'Expected dict or array-like, got {type(converted_dtypes)}')

        if copy:
            self._rs = self._rs.copy()
        self._rs = self._rs.astype(converted_dtypes)

    # Legacy methods that use as_type internally
    def as_dtype(self, dtypes: Iterable | pd.Index | dict[str, Dtype],
                 copy: bool = False) -> None:
        """
        Convert the ReflectionsCollection to a different data type.

        This method is deprecated. Use as_type instead.
        """
        self.as_type(dtypes, copy=copy)

    def as_mtz_types(self, mtz_types: Iterable[str], copy: bool = False) -> None:
        """
        Convert the ReflectionsCollection to a different MTZ data type.

        This method is deprecated. Use as_type instead.
        """
        if not isinstance(mtz_types, Iterable):
            raise TypeError(f'Expected iterable, got {type(mtz_types)}')
        if len(mtz_types) != self.no_columns:
            raise ValueError(f'Expected {self.no_columns} MTZ types, '
                             f'got {len(mtz_types)}')
        dtypes = []
        for i, mtz_type in enumerate(mtz_types):
            if mtz_type not in MTZ_COLUMN_TYPES:
                raise ValueError(f'Invalid MTZ type {mtz_type!r} for item {i}. '
                                 f'Choose one from: {", ".join(MTZ_COLUMN_TYPES)}')
            dtypes.append(MTZ_DTYPES[mtz_type])
        self.as_type(dtypes=dtypes, copy=copy)

    @property
    def mtz_types(self) -> tuple[str, ...]:
        """
        The MTZ types of the ReflectionsCollection
        """
        return tuple([dt.mtztype for dt in self.dtypes.to_list()])

    @property
    def columns(self) -> pd.Index:
        """
        The column names of the ReflectionsCollection
        """
        return self._rs.columns

    @property
    def labels(self) -> tuple[str, ...]:
        """
        The labels of the ReflectionsCollection
        """
        return tuple(self.columns.to_list())

    @property
    def values(self) -> rs.DataSet:
        """
        The data of the ReflectionsCollection
        """
        return self._rs

    # ToDo: Implement __getitem__ and __setitem__ for multiple columns
    # def __getitem__(self, item):
    #     return self._rs[self.label][item]
    #
    # def __setitem__(self, key, value) -> None:
    #     self._rs.loc[key, self.label] = value

    @classmethod
    def from_rs(cls, dataset: rs.DataSet, labels: Iterable[str] = None,
                space_group: gemmi.SpaceGroup | str | int = None,
                unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                           tuple[float | int] = None,
                merged: bool = None,
                metadata: ReflectionsMetadataType = None):
        """
        Create a ReflectionsCollection from a reciprocalspaceship DataSet.
        """
        if not isinstance(dataset, rs.DataSet):
            raise TypeError(f'Expected reciprocalspaceship.DataSet, got {type(dataset)}')
        # Extract space group, unit-cell and merged from rs.DataSet if not provided
        space_group = space_group or dataset.spacegroup
        unit_cell = unit_cell or dataset.cell
        merged = merged or dataset.merged

        if labels is not None:
            # Check if labels are present in dataset
            if not isinstance(labels, Iterable):
                raise TypeError(f'Expected iterable, got {type(labels)}')
            for i, label in enumerate(labels):
                if label not in dataset.columns:
                    raise ValueError(f'Label[{i}] {label!r} not found in DataSet columns. '
                                     f'Choose one from: {", ".join(dataset.columns)}')

            # Filter dataset to only include the labels
            dataset = dataset[labels]

        return cls(data=dataset, index=dataset.index, columns=dataset.columns,
                   types=dataset.dtypes, copy=True, space_group=space_group,
                   unit_cell=unit_cell, merged=merged, metadata=metadata)

    @classmethod
    def _from_rs_io(cls, file_type: str, file_reader: callable,
                    file: Path | str, labels: Iterable[str] = None,
                    space_group: gemmi.SpaceGroup | str | int = None,
                    unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                               tuple[float | int] = None,
                    merged: bool = None,
                    metadata: ReflectionsMetadataType = None):
        """
        Create a ReflectionsCollection from a reciprocalspaceship reader function.
        """
        # Check if file exists
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f'{file_type} file {file} not found')
        # Read the file
        dataset: rs.DataSet = file_reader(str(file))
        return cls.from_rs(dataset=dataset, labels=labels, space_group=space_group,
                           unit_cell=unit_cell, merged=merged, metadata=metadata)

    @classmethod
    def from_mtz(cls, mtz_file: Path | str, labels: Iterable[str] = None,
                 space_group: gemmi.SpaceGroup | str | int = None,
                 unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                            tuple[float | int] = None,
                 merged: bool = None,
                 metadata: ReflectionsMetadataType = None):
        """
        Create a ReflectionsCollection from an MTZ file and a column label.
        """
        return cls._from_rs_io(file_type='MTZ', file_reader=rs.read_mtz,
                               file=mtz_file, labels=labels, space_group=space_group,
                               unit_cell=unit_cell, merged=merged, metadata=metadata)

    @classmethod
    def from_cif(cls, cif_file: Path | str, labels: Iterable[str] = None,
                 space_group: gemmi.SpaceGroup | str | int = None,
                 unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                            tuple[float | int] = None,
                 merged: bool = None,
                 metadata: ReflectionsMetadataType = None):
        """
        Create a ReflectionsCollection from an CIF file and a column label.
        """
        return cls._from_rs_io(file_type='CIF', file_reader=rs.read_cif,
                               file=cif_file, labels=labels, space_group=space_group,
                               unit_cell=unit_cell, merged=merged, metadata=metadata)
