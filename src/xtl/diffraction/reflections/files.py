import abc
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

import gemmi
import reciprocalspaceship as rs

from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.options import Option, Options
from xtl.files.meta import FileContainer, FileReaderMeta

if TYPE_CHECKING:
    from xtl.diffraction.reflections import ReflectionsCollection

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


class ReflectionsFile(FileContainer[ReflectionsFileType, 'ReflectionsCollection'],
                      abc.ABC):
    """
    Base class for reading diffraction reflections files.
    """
    file_type: ReflectionsFileType
    """Type of reflections file"""


class ReflectionsFileReaders(FileReaderMeta[ReflectionsFileType, ReflectionsFile,
                                            'ReflectionsCollection']):
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


class MTZReflectionsFile(ReflectionsFile, metaclass=ReflectionsFileReaders):
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

    def read(self) -> 'ReflectionsCollection':
        """
        Read the MTZ file using gemmi.
        """
        from xtl.diffraction.reflections import ReflectionsCollection

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

    def read(self, block_id: int | str = 0) -> 'ReflectionsCollection':
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
        from xtl.diffraction.reflections import ReflectionsCollection

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

_mtz_summary = rs.summarize_mtz_dtypes(print_summary=False)
MTZ_DTYPES = {_mtz_summary['MTZ Code'][i]: getattr(rs.dtypes, _mtz_summary['Class'][i])
              for i in range(_mtz_summary.shape[0])}
"""Dictionary of MTZ column types to their corresponding ``rs.MTZDtype class``"""

MTZ_COLUMN_TYPES = set(MTZ_DTYPES)
"""Set of valid column types according to the MTZ specification."""


class GemmiCIF2MTZSpec(Options):
    """
    Specification for converting CIF to MTZ using Gemmi.
    """
    # https://github.com/project-gemmi/gemmi/blob/master/include/gemmi/cif2mtz.hpp#L146
    tag: str = Option(default=None, desc='CIF tag without category')
    column_label: str = Option(default=None, desc='MTZ column label')
    column_type: str = Option(default=None, choices=MTZ_COLUMN_TYPES,
                              desc='MTZ column type')
    dataset_id: int | None = Option(default=None, ge=0, le=1, desc='Dataset ID')
    map_fmt: str | None = Option(default=None, desc='Mapping instructions between '
                                                    'mmCIF symbols and MTZ numbers')

    @property
    def line(self) -> str:
        """
        Returns the specification line as a string.
        """
        l = f'{self.tag} {self.column_label} {self.column_type} {self.dataset_id}'
        if self.map_fmt:
            l += f' {self.map_fmt}'
        return l

    @classmethod
    def from_line(cls, line: str):
        """
        Create a spec from a string.
        """
        parts = line.split()
        if len(parts) == 4:
            tag, column_label, column_type, dataset_id = parts
            map_fmt = None
        elif len(parts) == 5:
            tag, column_label, column_type, dataset_id, map_fmt = parts
        else:
            raise ValueError(f'Invalid spec line: {line!r}, expected 4 or 5 parts, '
                             f'got {len(parts)}')
        return cls(
            tag=tag,
            column_label=column_label,
            column_type=column_type,
            dataset_id=int(dataset_id),
            map_fmt=map_fmt
        )


class GemmiMTZ2CIFSpec(Options):
    """
    Specification for converting MTZ to CIF using Gemmi.
    """
    # https://github.com/project-gemmi/gemmi/blob/master/include/gemmi/mtz2cif.hpp#L43
    column_label: str | None = Option(default=None, desc='MTZ column label')
    column_type: str | None = Option(default=None,
                                     choices=MTZ_COLUMN_TYPES | {'*', None},
                                     desc='MTZ column type')
    tag: str = Option(default=None, desc='CIF tag without category')
    flag: str | None = Option(default=None, choices=('?', '&', None),
                              desc='?: Ignore if column label not found,'
                                   '&: Ignore if previous the previous line was '
                                   'ignored')
    fmt: str | None = Option(default=None, regex=r'([#_+-]?\d*(\.\d+)?[fFgGeEc])|S',
                             desc='Float formatting string, similar to Python\'s '
                                  'format strings')
    special: str | None = Option(default=None, choices=('$counter', '$dataset',
                                                        '$image', '$.', '$?', None),
                                 desc='Special variables')

    @property
    def line(self) -> str:
        """
        Returns the specification line as a string.
        """
        if self.special is not None:
            return f'{self.special} {self.tag}'
        else:
            l = f'{self.column_label} {self.column_type} {self.tag}'
            if self.flag:
                l = f'{self.flag} {l}'
            if self.fmt:
                l += f' {self.fmt}'
            return l

    @classmethod
    def from_line(cls, line: str):
        """
        Create a spec from a string.
        """
        parts = line.split()
        column_label, column_type, tag = None, None, None
        flag, fmt, special = None, None, None
        if len(parts) == 2:
            special, tag = parts
        elif len(parts) == 3:
            column_label, column_type, tag = parts
        elif len(parts) == 4:
            if parts[0] in '&?':
                flag, column_label, column_type, tag = parts
            else:
                column_label, column_type, tag, fmt = parts
        elif len(parts) == 5:
            flag, column_label, column_type, tag, fmt = parts
        else:
            raise ValueError(f'Invalid spec line: {line!r}, expected 2-5 parts, '
                             f'got {len(parts)}')
        return cls(
            column_label=column_label,
            column_type=column_type,
            tag=tag,
            flag=flag,
            fmt=fmt,
            special=special
        )


class GemmiSpecs:
    """
    Class to hold the specifications for converting CIF to MTZ using Gemmi.
    """

    class CIF2MTZ:
        """
        Specifications for converting CIF to MTZ.

        Note that conversion of H, K, L, M/ISYM and BATCH is hardcoded and not part of
        the spec.
        """

        merged: tuple[GemmiCIF2MTZSpec, ...] = (
            GemmiCIF2MTZSpec.from_line('pdbx_r_free_flag FreeR_flag I 0'),
            GemmiCIF2MTZSpec.from_line('status FreeR_flag I 0 o=1,f=0'),
            GemmiCIF2MTZSpec.from_line('intensity_meas IMEAN J 1'),
            GemmiCIF2MTZSpec.from_line('F_squared_meas IMEAN J 1'),
            GemmiCIF2MTZSpec.from_line('intensity_sigma SIGIMEAN Q 1'),
            GemmiCIF2MTZSpec.from_line('F_squared_sigma SIGIMEAN Q 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_I_plus I(+) K 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_I_plus_sigma SIGI(+) M 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_I_minus I(-) K 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_I_minus_sigma SIGI(-) M 1'),
            GemmiCIF2MTZSpec.from_line('F_meas FP F 1'),
            GemmiCIF2MTZSpec.from_line('F_meas_au FP F 1'),
            GemmiCIF2MTZSpec.from_line('F_meas_sigma SIGFP Q 1'),
            GemmiCIF2MTZSpec.from_line('F_meas_sigma_au SIGFP Q 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_F_plus F(+) G 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_F_plus_sigma SIGF(+) L 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_F_minus F(-) G 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_F_minus_sigma SIGF(-) L 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_anom_difference DP D 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_anom_difference_sigma SIGDP Q 1'),
            GemmiCIF2MTZSpec.from_line('F_calc FC F 1'),
            GemmiCIF2MTZSpec.from_line('F_calc_au FC F 1'),
            GemmiCIF2MTZSpec.from_line('phase_calc PHIC P 1'),
            GemmiCIF2MTZSpec.from_line('fom FOM W 1'),
            GemmiCIF2MTZSpec.from_line('weight FOM W 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_HL_A_iso HLA A 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_HL_B_iso HLB A 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_HL_C_iso HLC A 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_HL_D_iso HLD A 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_FWT FWT F 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_PHWT PHWT P 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_DELFWT DELFWT F 1'),
            GemmiCIF2MTZSpec.from_line('pdbx_DELPHWT PHDELWT P 1')
        )
        """Specifications for merged data."""

        unmerged: tuple[GemmiCIF2MTZSpec, ...] = (
            GemmiCIF2MTZSpec.from_line('intensity_meas I J 0'),
            GemmiCIF2MTZSpec.from_line('intensity_net I J 0'),
            GemmiCIF2MTZSpec.from_line('intensity_sigma SIGI Q 0'),
            GemmiCIF2MTZSpec.from_line('pdbx_detector_x XDET R 0'),
            GemmiCIF2MTZSpec.from_line('pdbx_detector_y YDET R 0'),
            GemmiCIF2MTZSpec.from_line('pdbx_scan_angle ROT R 0')
        )
        """Specifications for unmerged data."""


    class MTZ2CIF:
        """
        Specifications for converting MTZ to CIF.
        """

        merged: tuple[GemmiMTZ2CIFSpec, ...]  = (
            GemmiMTZ2CIFSpec.from_line('H H index_h'),
            GemmiMTZ2CIFSpec.from_line('K H index_k'),
            GemmiMTZ2CIFSpec.from_line('L H index_l'),
            GemmiMTZ2CIFSpec.from_line('? FREE|RFREE|FREER|FreeR_flag|R-free-flags|'
                                       'FreeRflag I status S'),  # `S` -> status col
            GemmiMTZ2CIFSpec.from_line('? IMEAN|I|IOBS|I-obs J intensity_meas'),
            GemmiMTZ2CIFSpec.from_line('& SIG{prev} Q intensity_sigma'),
            GemmiMTZ2CIFSpec.from_line('? I(+)|IOBS(+)|I-obs(+)|Iplus K pdbx_I_plus'),
            GemmiMTZ2CIFSpec.from_line('& SIG{prev} M pdbx_I_plus_sigma'),
            GemmiMTZ2CIFSpec.from_line('? I(-)|IOBS(-)|I-obs(-)|Iminus K pdbx_I_minus'),
            GemmiMTZ2CIFSpec.from_line('& SIG{prev} M pdbx_I_minus_sigma'),
            GemmiMTZ2CIFSpec.from_line('? F|FP|FOBS|F-obs F F_meas_au'),
            GemmiMTZ2CIFSpec.from_line('& SIG{prev} Q F_meas_sigma_au'),
            GemmiMTZ2CIFSpec.from_line('? F(+)|FOBS(+)|F-obs(+)|Fplus G pdbx_F_plus'),
            GemmiMTZ2CIFSpec.from_line('& SIG{prev} L pdbx_F_plus_sigma'),
            GemmiMTZ2CIFSpec.from_line('? F(-)|FOBS(-)|F-obs(-)|Fminus G pdbx_F_minus'),
            GemmiMTZ2CIFSpec.from_line('& SIG{prev} L pdbx_F_minus_sigma'),
            GemmiMTZ2CIFSpec.from_line('? DP D pdbx_anom_difference'),
            GemmiMTZ2CIFSpec.from_line('& SIGDP Q pdbx_anom_difference_sigma'),
            GemmiMTZ2CIFSpec.from_line('? FC F F_calc'),
            GemmiMTZ2CIFSpec.from_line('? PHIC P phase_calc'),
            GemmiMTZ2CIFSpec.from_line('? FOM W fom'),
            GemmiMTZ2CIFSpec.from_line('? HLA A pdbx_HL_A_iso'),
            GemmiMTZ2CIFSpec.from_line('& HLB A pdbx_HL_B_iso'),
            GemmiMTZ2CIFSpec.from_line('& HLC A pdbx_HL_C_iso'),
            GemmiMTZ2CIFSpec.from_line('& HLD A pdbx_HL_D_iso'),
            GemmiMTZ2CIFSpec.from_line('? FWT|2FOFCWT F pdbx_FWT'),
            GemmiMTZ2CIFSpec.from_line('& PHWT|PH2FOFCWT P pdbx_PHWT .3f'),
            GemmiMTZ2CIFSpec.from_line('? DELFWT|FOFCWT F pdbx_DELFWT'),
            GemmiMTZ2CIFSpec.from_line('& DELPHWT|PHDELWT|PHFOFCWT P pdbx_DELPHWT .3f')
        )
        """Specifications for merged data."""

        unmerged: tuple[GemmiMTZ2CIFSpec, ...] = (
            GemmiMTZ2CIFSpec.from_line('$dataset diffrn_id'),
            GemmiMTZ2CIFSpec.from_line('$counter id'),
            GemmiMTZ2CIFSpec.from_line('H H index_h'),
            GemmiMTZ2CIFSpec.from_line('K H index_k'),
            GemmiMTZ2CIFSpec.from_line('L H index_l'),
            GemmiMTZ2CIFSpec.from_line('? I J intensity_net'),
            GemmiMTZ2CIFSpec.from_line('& SIGI Q intensity_sigma .5g'),
            GemmiMTZ2CIFSpec.from_line('? ROT R pdbx_scan_angle'),
            GemmiMTZ2CIFSpec.from_line('$image pdbx_image_id'),
        )
        """Specifications for unmerged data."""

