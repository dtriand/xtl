from typing import Any

import gemmi

from xtl.common.options import Option, Options
from xtl.common.serializers import GemmiUnitCell, GemmiSpaceGroup, GemmiMat33
from xtl.diffraction.reflections.files import ReflectionsFileType


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

    @property
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


class CIFDatasetMetadata(Options):
    wavelength: float | None = Option(default=None, gt=0.0,
                                      desc='Wavelength in Angstroms')
    unit_cell: gemmi.UnitCell | None = Option(default=None, formatter=GemmiUnitCell,
                                              desc='Unit cell parameters in '
                                                   'Angstroms/degrees')


class CIFReflectionsMetadata(ReflectionsMetadata):
    origin_file_type: ReflectionsFileType = Option(default=ReflectionsFileType.CIF)
    datasets: list[CIFDatasetMetadata] | None = Option(default=None)
    title: str | None = Option(default=None, max_length=64,
                               desc='Title of the CIF file')
    spec_lines: list[str] | None = Option(default=None,
                                          desc='List of specification lines')
    history: list[str] | None = Option(default=None,
                                       desc='History of the CIF file')


ReflectionsMetadataType = ReflectionsMetadata | MTZReflectionsMetadata | \
                          CIFReflectionsMetadata