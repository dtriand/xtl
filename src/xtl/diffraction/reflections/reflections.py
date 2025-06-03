from pathlib import Path
from typing import Optional, Iterable, Union

import gemmi
import reciprocalspaceship as rs
from reciprocalspaceship.dtypes.base import MTZDtype
import numpy as np
import pandas as pd
from pandas._typing import Dtype, ArrayLike


from xtl.diffraction.reflections.mtz_types import mtz_types as _mtz_types
from xtl.diffraction.reflections.metadata import *


class Reflection:
    # Representation of a single reflection with one or more columns of data
    ...


class ReflectionsData:

    def __init__(self,
                 # pd.DataFrame arguments
                 data: ArrayLike | Iterable | dict | pd.DataFrame | rs.DataSeries |
                       gemmi.Mtz,
                 index: Optional[ArrayLike | pd.Index] = None,
                 columns: Optional[ArrayLike | pd.Index] = None,
                 dtype: Optional[Dtype] = None,
                 copy: bool = False,
                 # Additional rs.DataSet arguments
                 space_group: Optional[gemmi.SpaceGroup] = None,
                 unit_cell: Optional[gemmi.UnitCell] = None,
                 merged: Optional[bool] = None
                 ):
        # Extract space group and unit-cell from gemmi.Mtz if not provided
        if isinstance(data, gemmi.Mtz):
            space_group = space_group or data.spacegroup
            unit_cell = unit_cell or data.cell
        self._rs = rs.DataSet(data=data, index=index, columns=columns, dtype=dtype,
                              copy=copy, spacegroup=space_group, cell=unit_cell,
                              merged=merged)
        if self._rs.shape[1] != 1:
            raise ValueError(f'{self.__class__.__name__} must have 1 column, but got '
                             f'{self._rs.shape[1]} columns')

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the ReflectionsData
        """
        return self._rs.shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions
        """
        return self._rs.ndim

    @property
    def no_reflections(self) -> int:
        """
        The number of reflections in the ReflectionsData
        """
        return self.shape[0]

    @property
    def no_columns(self) -> int:
        """
        The number of columns in the ReflectionsData
        """
        return self.shape[1]

    @property
    def dtype(self) -> MTZDtype:
        """
        The data type of the ReflectionsData
        """
        return self._rs.dtypes.iloc[0]

    def as_dtype(self, dtype: MTZDtype, copy: bool = False) -> None:
        """
        Convert the ReflectionsData to a different data type.
        """
        if copy:
            self._rs = self._rs.copy()
        self._rs = self._rs.astype(dtype)

    @property
    def mtz_type(self) -> str:
        """
        The MTZ type of the ReflectionsData
        """
        return self.dtype.mtztype

    def as_mtz_type(self, mtz_type: str, copy: bool = False) -> None:
        """
        Convert the ReflectionsData to a different MTZ data type.
        """
        if mtz_type not in _mtz_types.keys():
            raise ValueError(f'Invalid MTZ type {mtz_type!r}. '
                             f'Choose one from: {", ".join(list(_mtz_types.keys()))}')
        self.as_dtype(_mtz_types[mtz_type](), copy=copy)

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
    def hkls(self) -> np.ndarray:
        """
        An array of the HKL indices
        """
        return self._rs.get_hkls()

    @property
    def values(self) -> rs.DataSeries:
        """
        The data of the ReflectionsData
        """
        return self._rs[self.label]

    @property
    def space_group(self) -> gemmi.SpaceGroup:
        """
        The space group of the ReflectionsData
        """
        return self._rs.spacegroup

    @space_group.setter
    def space_group(self, space_group: gemmi.SpaceGroup | str | int):
        """
        Set the space group of the ReflectionsData
        """
        self._rs.spacegroup = space_group

    @property
    def unit_cell(self) -> gemmi.UnitCell:
        """
        The unit-cell of the ReflectionsData
        """
        return self._rs.cell

    @unit_cell.setter
    def unit_cell(self, unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                                   tuple[float | int]):
        """
        Set the unit-cell of the ReflectionsData
        """
        self._rs.cell = unit_cell

    @property
    def merged(self) -> bool:
        """
        The merged status of the ReflectionsData
        """
        return self._rs.merged

    @merged.setter
    def merged(self, merged: bool):
        """
        Set the merged status of the ReflectionsData
        """
        if not isinstance(merged, bool) and merged is not None:
            raise TypeError(f'Must be a boolean, got {type(merged)}')
        self._rs.merged = merged

    def __getitem__(self, item):
        # ToDo: More sophisticated implementation needed
        return self._rs[self.label][item]

    def __setitem__(self, key, value) -> None:
        self._rs.loc[key, self.label] = value

    @classmethod
    def from_rs(cls, dataset: rs.DataSet, label: str,
                space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                    tuple[float | int]] = None,
                merged: Optional[bool] = None):
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
                   columns=dataset.columns[ic:ic+1], dtype=dataset.dtypes[label],
                   copy=True, space_group=space_group, unit_cell=unit_cell,
                   merged=merged)

    @classmethod
    def _from_rs_io(cls, file_type: str, file_reader: callable,
                    file: Path | str, label: str,
                    space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                    unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                    tuple[float | int]] = None,
                    merged: Optional[bool] = None):
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
                 space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                 unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                     tuple[float | int]] = None,
                 merged: Optional[bool] = None):
        """
        Create a ReflectionsData from an MTZ file and a column label.
        """
        return cls._from_rs_io(file_type='MTZ', file_reader=rs.read_mtz,
                                file=mtz_file, label=label, space_group=space_group,
                                unit_cell=unit_cell, merged=merged)

    @classmethod
    def from_cif(cls, cif_file: Path | str, label: str,
                 space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                 unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                     tuple[float | int]] = None,
                 merged: Optional[bool] = None):
        """
        Create a ReflectionsData from an CIF file and a column label.
        """
        return cls._from_rs_io(file_type='CIF', file_reader=rs.read_cif,
                                file=cif_file, label=label, space_group=space_group,
                                unit_cell=unit_cell, merged=merged)



class ReflectionsCollection:

    def __init__(self,
                 # pd.DataFrame arguments
                 data: ArrayLike | Iterable | dict | pd.DataFrame | rs.DataSet |
                       gemmi.Mtz,
                 index: Optional[ArrayLike | pd.Index] = None,
                 columns: Optional[ArrayLike | pd.Index] = None,
                 dtypes: Optional[Iterable | pd.Index | pd.Series | dict[str, Dtype]] = None,
                 copy: bool = False,
                 # Additional rs.DataSet arguments
                 space_group: Optional[gemmi.SpaceGroup] = None,
                 unit_cell: Optional[gemmi.UnitCell] = None,
                 merged: Optional[bool] = None,
                 # XTL arguments
                 metadata: Optional[ReflectionsMetadataType] = None
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
        if dtypes is not None:
            self.as_dtype(dtypes)

    @property
    def metadata(self) -> Optional[ReflectionsMetadataType]:
        """
        Metadata associated with the reflections when instantiated from a file.
        """
        return self._metadata

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the ReflectionsCollection
        """
        return self._rs.shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions
        """
        return self._rs.ndim

    @property
    def no_reflections(self) -> int:
        """
        The number of reflections in the ReflectionsCollection
        """
        return self.shape[0]

    @property
    def no_columns(self) -> int:
        """
        The number of columns in the ReflectionsCollection
        """
        return self.shape[1]

    @property
    def dtypes(self) -> pd.Series:
        """
        The data types of the ReflectionsCollection
        """
        return self._rs.dtypes

    def as_dtype(self, dtypes: Iterable | pd.Index | dict[str, Dtype],
                 copy: bool = False) -> None:
        if isinstance(dtypes, dict):
            if len(dtypes.keys()) != self.no_columns:
                raise ValueError(f'Expected {self.no_columns} dtypes, '
                                 f'got {len(dtypes.keys())}')
        elif isinstance(dtypes, Iterable | pd.Index):
            if len(dtypes) != self.no_columns:
                raise ValueError(f'Expected {self.no_columns} dtypes, got {len(dtypes)}')
            dtypes = {col: dt for col, dt in zip(self.columns, dtypes)}
        else:
            raise TypeError(f'Expected dict or array-like, got {type(dtypes)}')
        if copy:
            self._rs = self._rs.copy()
        self._rs = self._rs.astype(dtypes)

    @property
    def mtz_types(self) -> tuple[str, ...]:
        """
        The MTZ types of the ReflectionsCollection
        """
        return tuple([dt.mtztype for dt in self.dtypes.to_list()])

    def as_mtz_types(self, mtz_types: Iterable[str], copy: bool = False) -> None:
        """
        Convert the ReflectionsCollection to a different MTZ data type.
        """
        if not isinstance(mtz_types, Iterable):
            raise TypeError(f'Expected iterable, got {type(mtz_types)}')
        if len(mtz_types) != self.no_columns:
            raise ValueError(f'Expected {self.no_columns} MTZ types, '
                             f'got {len(mtz_types)}')
        dtypes = []
        for i, mtz_type in enumerate(mtz_types):
            if mtz_type not in _mtz_types.keys():
                raise ValueError(f'Invalid MTZ type {mtz_type!r} for item {i}. '
                                 f'Choose one from: {", ".join(list(_mtz_types.keys()))}')
            dtypes.append(_mtz_types[mtz_type]())
        self.as_dtype(dtypes=dtypes, copy=copy)

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
    def hkls(self) -> np.ndarray:
        """
        An array of the HKL indices
        """
        return self._rs.get_hkls()

    @property
    def values(self) -> rs.DataSet:
        """
        The data of the ReflectionsCollection
        """
        return self._rs

    @property
    def space_group(self) -> gemmi.SpaceGroup:
        """
        The space group of the ReflectionsCollection
        """
        return self._rs.spacegroup

    @space_group.setter
    def space_group(self, space_group: gemmi.SpaceGroup | str | int):
        """
        Set the space group of the ReflectionsCollection
        """
        self._rs.spacegroup = space_group

    @property
    def unit_cell(self) -> gemmi.UnitCell:
        """
        The unit-cell of the ReflectionsCollection
        """
        return self._rs.cell

    @unit_cell.setter
    def unit_cell(self, unit_cell: gemmi.UnitCell | np.ndarray | list[float | int] |
                                   tuple[float | int]):
        """
        Set the unit-cell of the ReflectionsCollection
        """
        self._rs.cell = unit_cell

    @property
    def is_merged(self) -> bool:
        """
        The merged status of the ReflectionsCollection
        """
        return self._rs.merged

    @is_merged.setter
    def is_merged(self, merged: bool):
        """
        Set the merged status of the ReflectionsCollection
        """
        if not isinstance(merged, bool) and merged is not None:
            raise TypeError(f'Must be a boolean, got {type(merged)}')
        self._rs.merged = merged

    # ToDo: Implement __getitem__ and __setitem__ for multiple columns
    # def __getitem__(self, item):
    #     return self._rs[self.label][item]
    #
    # def __setitem__(self, key, value) -> None:
    #     self._rs.loc[key, self.label] = value

    @classmethod
    def from_rs(cls, dataset: rs.DataSet, labels: Optional[Iterable[str]] = None,
                space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                    tuple[float | int]] = None,
                merged: Optional[bool] = None,
                metadata: Optional[ReflectionsMetadataType] = None):
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
                   dtypes=dataset.dtypes, copy=True, space_group=space_group,
                   unit_cell=unit_cell, merged=merged, metadata=metadata)

    @classmethod
    def _from_rs_io(cls, file_type: str, file_reader: callable,
                    file: Path | str, labels: Optional[Iterable[str]] = None,
                    space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                    unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                    tuple[float | int]] = None,
                    merged: Optional[bool] = None,
                    metadata: Optional[ReflectionsMetadataType] = None):
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
    def from_mtz(cls, mtz_file: Path | str, labels: Optional[Iterable[str]] = None,
                 space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                 unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                     tuple[float | int]] = None,
                 merged: Optional[bool] = None,
                 metadata: Optional[ReflectionsMetadataType] = None):
        """
        Create a ReflectionsCollection from an MTZ file and a column label.
        """
        return cls._from_rs_io(file_type='MTZ', file_reader=rs.read_mtz,
                                file=mtz_file, labels=labels, space_group=space_group,
                                unit_cell=unit_cell, merged=merged, metadata=metadata)

    @classmethod
    def from_cif(cls, cif_file: Path | str, labels: Optional[Iterable[str]] = None,
                 space_group: Optional[gemmi.SpaceGroup | str | int] = None,
                 unit_cell: Optional[gemmi.UnitCell | np.ndarray | list[float | int] |
                                     tuple[float | int]] = None,
                 merged: Optional[bool] = None,
                 metadata: Optional[ReflectionsMetadataType] = None):
        """
        Create a ReflectionsCollection from an CIF file and a column label.
        """
        return cls._from_rs_io(file_type='CIF', file_reader=rs.read_cif,
                                file=cif_file, labels=labels, space_group=space_group,
                                unit_cell=unit_cell, merged=merged, metadata=metadata)
