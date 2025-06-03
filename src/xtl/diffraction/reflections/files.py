import abc
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import gemmi

from xtl.common.compatibility import PY310_OR_LESS
from xtl.diffraction.reflections import ReflectionsCollection
from xtl.files.meta import FileContainer, FileReaderMeta

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class ReflectionsFileType(StrEnum):
    """
    Enum for different types of reflections files.
    """
    MTZ = 'mtz'
    CIF = 'mmcif'
    XDS_ASCII = 'xds_ascii'


class ReflectionsFile(FileContainer[ReflectionsFileType, ReflectionsCollection],
                      abc.ABC):
    """
    Base class for reading diffraction reflections files.
    """
    file_type: ReflectionsFileType
    """Type of reflections file"""


class ReflectionsFileReaders(FileReaderMeta[ReflectionsFileType, ReflectionsFile,
                                            ReflectionsCollection]):
    """
    Metaclass for registering diffraction reflections file types.
    This allows for dynamic registration of new file types.
    """
    extensions = {
        ReflectionsFileType.MTZ: ['.mtz'],
        ReflectionsFileType.CIF: ['.cif'],
        ReflectionsFileType.XDS_ASCII: ['.hkl']
    }
    base_class = ReflectionsFile


read_reflections = ReflectionsFileReaders.read_file
"""Generic reader for diffraction reflections files."""


class MTZReflectionsFile(ReflectionsFile):
    """
    Class for reading MTZ reflections files.
    """
    file_type = ReflectionsFileType.MTZ

    @staticmethod
    def _get_endianness_float(b: int) -> str | None:
        """
        Determine the endianness for floating point numbers based on the half-byte.
        """
        if b == 1:
            return '>'  # DFNTF_BEIEEE: Big-endian IEEE 754 format
        elif b == 2:
            return None  # DFNTF_VAX: Unsupported legacy format
        elif b == 4:
            return '<'  # DFNTF_LEIEEE: Little-endian IEEE 754 format
        elif b == 5:
            return None  # DFNTF_CONVEXNATIVE: Unsupported legacy format
        return None

    @staticmethod
    def _get_endianness_int(b: int) -> str | None:
        """
        Determine the endianness for integers based on the half-byte.
        """
        if b == 1:
            return '>'  # DFNTI_MBO: Motorola Byte Order  (big-endian)
        elif b == 4:
            return '<'  # DFNTI_IBO: Intel Byte Order (little-endian)
        return None

    @staticmethod
    def _read_header(b: bytes) -> dict[str, Any]:
        content = b.decode(encoding='ascii', errors='ignore')
        header = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('END'):
                # End of header
                break
            elif line.startswith('VERS'):
                # Version line
                header['version'] = line[4:].strip()
            elif line.startswith('TITLE'):
                # Title line
                header['title'] = line[5:].strip()
            elif line.startswith('NCOL'):
                ...

    @staticmethod
    def _parse_mtz(file: str | Path) -> bool:
        """
        Experimental and incomplete byte-level parser for MTZ files.
        """
        warnings.warn('This is an experimental MTZ parser, your mileage may vary...')
        import struct

        file = Path(file)
        with file.open('rb') as f:
            # Check magic string
            f.seek(0)
            magic_string = f.read(4)
            if magic_string != b'MTZ ':
                raise ValueError('No MTZ magic string found in file')

            # Decode machine stamp
            # NB: https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=CCP4BB;51ea3cae.1811
            f.seek(8)
            machine_stamp = f.read(2)  # 2 bytes or 4 half-bytes (nibbles)
            b0, b1 = struct.unpack('bb', machine_stamp)
            # Split the 2 bytes into 4 nibbles
            nibbles = [(b0 >> 4) & 0x0f, b0 & 0x0f, (b1 >> 4) & 0x0f, b1 & 0x0f]
            hb0, hb1, hb2, hb3 = nibbles
            endianness_float = MTZReflectionsFile._get_endianness_float(hb0)
            endianness_complex = MTZReflectionsFile._get_endianness_float(hb1)
            endianness_int = MTZReflectionsFile._get_endianness_int(hb2)
            # hb3 -> endianness_string / not used in this context
            if not endianness_float or not endianness_complex or not endianness_int:
                raise ValueError(
                    'Unsupported MTZ machine stamp: '
                    f'{machine_stamp.hex()} (nibbles: {nibbles})\n'
                    f'Endianness: float={endianness_float}, '
                    f'complex={endianness_complex}, int={endianness_int}'
                )
            # Check if the endianness is consistent
            if endianness_float != endianness_complex or \
                    endianness_float != endianness_int:
                raise ValueError(
                    f'Inconsistent endianness in machine stamp: '
                    f'{machine_stamp.hex()} -> {nibbles}\n'
                    f'Endianness: float={endianness_float}, '
                    f'complex={endianness_complex}, int={endianness_int}'
                )
            # Alias for endianness
            e = endianness_float

            # Get the header offset
            f.seek(4)
            header_offset = struct.unpack(f'{e}i', f.read(4))[0]
            f.seek(header_offset)
            header = MTZReflectionsFile._read_header(f.read())

    @staticmethod
    def sniff(file: str | Path) -> bool:
        """
        Check if the file is a valid MTZ file by looking for the magic string.
        """
        file = Path(file)
        with file.open('rb') as f:
            # Check magic string
            f.seek(0)
            magic_string = f.read(4)
            if magic_string != b'MTZ ':
                return False
        return True

    def read(self) -> ReflectionsCollection:
        """
        Read the MTZ file using gemmi.
        """
        mtz = gemmi.read_mtz_file(str(self.file))
        return ReflectionsCollection(data=mtz)


class CIFReflectionsFile(ReflectionsFile, metaclass=ReflectionsFileReaders):
    """
    Class for reading CIF reflections files.
    """
    file_type = ReflectionsFileType.CIF

    @staticmethod
    def sniff(file: str | Path) -> bool:
        """
        Check if a file is a valid structure factors CIF file by looking for:
        1. The presence of a data directive (starts with `data_`)
        2. The presence of reflection indices in the form of:
           - `_refln.index_h`, `_refln.index_k`, `_refln.index_l` (for merged data)
           - `_diffrn_refln.index_h`, `_diffrn_refln.index_k`, `_diffrn_refln.index_l` (for unmerged data)
        """
        file = Path(file)
        with file.open('r') as f:
            # Strip comments and white space
            lines = [line.replace('#', '').strip() for line in f.readlines()]

            # Check for data directive
            has_data_block = False
            for line in lines:
                if line.startswith('data_'):
                    has_data_block = True
                    break
            if not has_data_block:
                return False

            # Check for merged data
            has_h, has_k, has_l = False, False, False
            for line in lines:
                if all(has_h, has_k, has_l):
                    # Contains merged data
                    return True
                if line.startswith('_refln.index_h'):
                    has_h = True
                elif line.startswith('_refln.index_k'):
                    has_k = True
                elif line.startswith('_refln.index_l'):
                    has_l = True

            # Check for unmerged data
            has_h, has_k, has_l = False, False, False
            for line in lines:
                if all(has_h, has_k, has_l):
                    # Contains unmerged data
                    return True
                if line.startswith('_diffrn_refln.index_h'):
                    has_h = True
                elif line.startswith('_diffrn_refln.index_k'):
                    has_k = True
                elif line.startswith('_diffrn_refln.index_l'):
                    has_l = True

            # No reflection data found :(
            return False

    def read(self, block_id: int | str = 0) -> ReflectionsCollection:
        """
        Read the CIF file and return a ReflectionsCollection. If multiple data blocks
        are present in the CIF file, the `block_id` parameter can be used to select
        a specific one.

        :param block_id: Index or name of the data block to read (default: 0).
        :raise IndexError: If the specified `block_id` does not exist in the CIF file.
        :raise TypeError: If `block_id` is not an int or str.
        """
        import gemmi
        import reciprocalspaceship as rs

        cif = gemmi.cif.read(str(self.file))
        blocks = gemmi.as_refln_blocks(cif)

        if isinstance(block_id, str):
            # Get the index of the block by name
            block_names = [block.block.name for block in blocks]
            if block_id not in block_names:
                raise IndexError(f'Block {block_id!r} not found in CIF file. '
                                 f'Available blocks: {block_names}')
            block_id = block_names.index(block_id)
        elif not isinstance(block_id, int):
            raise TypeError(f'\'block_id\' must be an int or str, got {type(block_id)}')

        # Check if the block_id is within the range of available blocks
        if block_id >= len(blocks):
            raise IndexError(f'File {self.file} has only {len(blocks)} blocks, '
                             f'but {block_id=} was requested.')

        block = blocks[block_id]
        mtz = gemmi.CifToMtz().convert_block_to_mtz(block)
        ds = rs.io.from_gemmi(mtz)

        return ReflectionsCollection.from_rs(dataset=ds)
