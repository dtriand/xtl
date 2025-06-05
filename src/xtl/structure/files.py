import abc
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from xtl.common.compatibility import PY310_OR_LESS
from xtl.files.meta import FileContainer, FileReaderMeta

if TYPE_CHECKING:
    from xtl.structure.atoms import AtomsCollection

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class StructureFileType(StrEnum):
    """
    Enum for different types of atomic structure files.
    """
    PDB = 'pdb'
    CIF = 'mmcif'


class StructureFile(FileContainer[StructureFileType, 'AtomsCollection'],
                    abc.ABC):
    file_type: StructureFileType
    """Type of atomic structure file"""


class StructureFileReaders(FileReaderMeta[StructureFileType, StructureFile,
                                          'AtomsCollection']):
    """
    Metaclass for registering atomic structure file types.
    This allows for dynamic registration of new file types.
    """
    extensions = {
        StructureFileType.PDB: ['.pdb', '.ent'],
        StructureFileType.CIF: ['.cif', '.mmcif']
    }
    base_class = StructureFile


read_structure = StructureFileReaders.read_file
"""Generic reader for atomic structure files."""


class PDBStructureFile(StructureFile, metaclass=StructureFileReaders):
    """
    Class for reading structure files in PDB format.
    """
    file_type = StructureFileType.PDB

    @staticmethod
    def sniff(file: str | Path) -> bool:
        """
        Check if the file is a valid PDB file by looking for the presence of ATOM
        records.
        """
        file = Path(file)
        with file.open('r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    return True
        return False

    def read(self) -> 'AtomsCollection':
        raise NotImplementedError()


class CIFStructureFile(StructureFile, metaclass=StructureFileReaders):
    """
    Class for reading structure files in mmCIF format.
    """
    file_type = StructureFileType.CIF

    @staticmethod
    def sniff(file: str | Path) -> bool:
        """
        Check if the file is a valid atomic structure CIF file by looking for:
        1. The presence of a `data_` directive
        2. The presence of `_atom_site` records
        """
        file = Path(file)
        with file.open('r') as f:
            # Strip comments and whitespace
            lines = [line.replace('#', '').strip() for line in f]

            # Check for data directive
            has_data_block = False
            for line in lines:
                if line.startswith('data_'):
                    has_data_block = True
                    break
            if not has_data_block:
                return False

            # Check for atom site records
            for line in lines:
                if line.startswith('_atom_site'):
                    return True

        # No atom records found :(
        return False

    def read(self) -> 'AtomsCollection':
        raise NotImplementedError()
