__all__ = [
    'ReflectionsMetadata',
    'MTZBatchMetadata',
    'MTZDatasetMetadata',
    'MTZReflectionsMetadata',
    'CIFReflectionsMetadata',
    'ReflectionsMetadataType'
]

from typing import Any, Sequence, overload

import gemmi
from pydantic import computed_field

from xtl.common.options import Option, Options
from xtl.common.serializers import GemmiUnitCell, GemmiSpaceGroup, GemmiMat33
from xtl.diffraction.reflections.files import ReflectionsFileType, GemmiCIF2MTZSpec


class ReflectionsMetadata(Options):
    """
    Metadata for diffraction reflections that were extracted from a file.
    """

    origin_file_type: ReflectionsFileType | None = Option(default=None,
                      desc='Type of the file where the reflections originated from')
    name: str | None = Option(default=None, desc='Dataset title')
    wavelength: float | None = Option(default=None, ge=0.0,
                                      desc='Wavelength in Angstroms')
    unit_cell: gemmi.UnitCell | None = Option(default=None, formatter=GemmiUnitCell,
                                              desc='Unit cell in Angstroms/degrees')
    space_group: gemmi.SpaceGroup | None = Option(default=None,
                                                  formatter=GemmiSpaceGroup,
                                                  desc='Space group in Hermann-Mauguin '
                                                       'notation')
    resolution_high: float | None = Option(default=None, gt=0.0,
                                           desc='Reported high resolution limit in '
                                                'Angstroms')
    resolution_low: float | None = Option(default=None, gt=0.0,
                                          desc='Reported low resolution limit in '
                                               'Angstroms')
    no_reflections: int | None = Option(default=None, ge=0,
                                        desc='Reported number of reflections')
    column_labels: tuple[str, ...] | None = Option(default=None,
                                                   desc='Column labels')
    column_types: tuple[str, ...] | None = Option(default=None,
                                                  desc='MTZ column types')


class MTZBatchMetadata(Options):
    """
    Metadata for a batch in an MTZ file.
    """

    id: int | None = Option(default=None, desc='Batch ID')
    name: str | None = Option(default=None, max_length=64, desc='Batch title')
    wavelength: float | None = Option(default=None, ge=0.0,
                                      desc='Wavelength in Angstroms')
    unit_cell: gemmi.UnitCell | None = Option(default=None, formatter=GemmiUnitCell,
                                              desc='Unit cell in Angstroms/degrees')
    axes: tuple[str, ...] | None = Option(default=None, desc='Rotation axes')
    phi_start: float | None = Option(default=None,
                                     desc='Oscillation start angle in degrees')
    phi_end: float | None = Option(default=None,
                                   desc='Oscillation end angle in degrees')
    phi_step: float | None = Option(default=None, desc='Oscillation step in degrees')
    matrix_U: gemmi.Mat33 | None = Option(default=None, formatter=GemmiMat33,
                                          desc='Crystal orientation U matrix')
    mosaicity: float | None = Option(default=None, gt=0.0,
                                     desc='Batch mosaicity in degrees')

    @classmethod
    def from_gemmi(cls, batch: gemmi.Mtz.Batch):
        """
        Create an instance from a ``gemmi.Mtz.Batch`` object.
        """
        # Deciphered from:
        # - gemmi.Mtz.Batch: https://github.com/project-gemmi/gemmi/blob/master/include/gemmi/mtz.hpp
        # - mtzdump HKLIN <filename> BATCH END
        # But more information is stored in batch.ints and batch.floats
        return cls(
            id=batch.number,
            title=batch.title.replace('TITLE ', '').strip(),
            wavelength=batch.floats[86],
            unit_cell=gemmi.UnitCell(*batch.floats[0:6]),
            axes=tuple(batch.axes),
            phi_start=batch.floats[36],
            phi_end=batch.floats[37],
            phi_step=batch.floats[47],
            matrix_U=gemmi.Mat33([[batch.floats[i + j] for i in [6, 9, 12]]
                                   for j in range(3)]),
            mosaicity=batch.floats[21]
        )


class MTZDatasetMetadata(Options):
    """
    Metadata for a dataset in an MTZ file.
    """

    id: int | None = Option(default=None, desc='Dataset ID')
    project_name: str | None = Option(default=None, max_length=64,
                                      desc='MTZ project name')
    crystal_name: str | None = Option(default=None, max_length=64,
                                      desc='MTZ crystal name')
    dataset_name: str | None = Option(default=None, max_length=64,
                                      desc='MTZ dataset name')
    wavelength: float | None = Option(default=None, ge=0.0,
                                      desc='Wavelength in Angstroms')
    unit_cell: gemmi.UnitCell | None = Option(default=None, formatter=GemmiUnitCell,
                                              desc='Unit cell parameters in '
                                                   'Angstroms/degrees')

    @property
    def tree(self) -> str:
        """
        The hierarchical tree structure of the dataset
        """
        return f'{self.project_name}/{self.crystal_name}/{self.dataset_name}'

    @property
    def is_base(self) -> bool:
        """
        Check if this is the base dataset of the MTZ file (according to CCP4 5.0)
        """
        return (self.tree == 'HKL_base/HKL_base/HKL_base') and (self.id == 0)

    def __repr__(self):
        return f'{self.__class__.__name__} (id={self.id}, {self.tree})'

    @classmethod
    def from_gemmi(cls, dataset: gemmi.Mtz.Dataset):
        """
        Create an instance from a ``gemmi.Mtz.Dataset`` object.
        """
        return cls(
            id=dataset.id,
            project_name=dataset.project_name,
            crystal_name=dataset.crystal_name,
            dataset_name=dataset.dataset_name,
            wavelength=dataset.wavelength,
            unit_cell=dataset.cell
        )


class MTZReflectionsMetadata(ReflectionsMetadata):
    """
    Metadata for reflections extracted from an MTZ file.
    """

    origin_file_type: ReflectionsFileType = Option(default=ReflectionsFileType.MTZ,
                                                   desc='Type of the file where the '
                                                        'reflections originated from')
    datasets: tuple[MTZDatasetMetadata, ...] | None = Option(default=None,
                                                             desc='Metadata for datasets'
                                                                  ' in the MTZ file')
    batches: tuple[MTZBatchMetadata, ...] | None = Option(default=None,
                                                          desc='Metadata for batches '
                                                               'in the MTZ file')
    missing_value: Any | None = Option(default=None, desc='Value that represents '
                                                          'missing values (VALM)')
    history: tuple[str, ...] | None = Option(default=None, desc='History lines')

    @computed_field(description='Whether the reflections are merged or unmerged')
    def is_merged(self) -> bool | None:
        """
        Check if the reflections are merged or unmerged. The presence of batches in the
        metadata will be considered as an indication of unmerged data.
        """
        if self.batches is None:
            return None
        elif self.batches:
            return False
        return True

    @classmethod
    def from_gemmi(cls, mtz: gemmi.Mtz):
        """
        Create an instance from a ``gemmi.Mtz`` object.
        """
        return cls(
            origin_file_type=ReflectionsFileType.MTZ,
            name=mtz.title,
            unit_cell=mtz.cell,
            space_group=mtz.spacegroup,
            resolution_high=mtz.resolution_high(),
            resolution_low=mtz.resolution_low(),
            no_reflections=mtz.nreflections,
            column_labels=tuple(mtz.column_labels()),
            column_types=tuple(column.type for column in mtz.columns),
            datasets=tuple(MTZDatasetMetadata.from_gemmi(ds) for ds in mtz.datasets),
            batches=tuple(MTZBatchMetadata.from_gemmi(batch) for batch in mtz.batches),
            missing_value=mtz.valm,
            history=tuple(mtz.history)
        )


class CIFReflectionsMetadata(ReflectionsMetadata):
    """
    Metadata for reflections extracted from a CIF file.
    """

    origin_file_type: ReflectionsFileType = Option(default=ReflectionsFileType.CIF,
                                                   desc='Type of the file where the '
                                                        'reflections originated from')
    entry_id: str | None = Option(default=None, desc='Entry ID of the CIF block')
    is_merged: bool | None = Option(default=None, desc='Whether the reflections are '
                                                       'merged or unmerged')
    spec_lines: tuple[GemmiCIF2MTZSpec, ...] | None = \
        Option(default_factory=tuple,
               formatter=lambda x: tuple(l.line for l in x),
               desc='List of specification lines used during CIF to MTZ conversion')

    @computed_field(description='Column types inferred from the column labels using '
                                'the specification lines, if available')
    def column_types_inferred(self) -> tuple[str | None, ...] | None:
        """
        Infer the column types from the column labels using the specification lines,
        if available.
        """
        column_types = []
        if not self.column_labels or not self.spec_lines:
            return None

        specs = list(self.spec_lines) + \
                [GemmiCIF2MTZSpec.from_line(l) for l in
                 ['index_h H H 0', 'index_k K H 0', 'index_l L H 0']]
        for label in self.column_labels:
            inferred = False
            for spec in specs:
                if label == spec.tag:
                    column_types.append(spec.column_type)
                    inferred = True
                    break
            if not inferred:
                column_types.append(None)

        return tuple(column_types)

    @classmethod
    def from_gemmi(cls, rblock: gemmi.ReflnBlock,
                   spec_lines: Sequence[str | GemmiCIF2MTZSpec] = None):
        """
        Create an instance from a ``gemmi.ReflnBlock`` object.
        """
        if not isinstance(rblock, gemmi.ReflnBlock):
            raise TypeError(f'Expected gemmi.ReflnBlock, got {type(rblock).__name__}')

        resolution_high = rblock.block.find_value('_reflns.d_resolution_high')
        resolution_low = rblock.block.find_value('_reflns.d_resolution_low')
        n_obs = rblock.block.find_value('_reflns.number_obs')
        resolution_high = float(resolution_high) if resolution_high else None
        resolution_low = float(resolution_low) if resolution_low else None
        n_obs = int(n_obs) if n_obs else None

        return cls(
            origin_file_type=ReflectionsFileType.CIF,
            name=rblock.block.name,
            entry_id=rblock.entry_id,
            unit_cell=rblock.cell,
            space_group=rblock.spacegroup,
            wavelength=rblock.wavelength,
            resolution_high=resolution_high,
            resolution_low=resolution_low,
            no_reflections=n_obs,
            is_merged=rblock.is_merged(),
            column_labels=tuple(rblock.column_labels()),
            spec_lines=spec_lines or tuple()
        )



ReflectionsMetadataType = ReflectionsMetadata | MTZReflectionsMetadata | \
                          CIFReflectionsMetadata