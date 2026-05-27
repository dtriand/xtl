import abc
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
from typing import TYPE_CHECKING

from xtl import Logger
from xtl.common.compatibility import PY310_OR_LESS
from xtl.files.meta import FileContainer, FileReaderMeta
if TYPE_CHECKING:
    from xtl.scattering.profiles import ScatteringProfile

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


logger = Logger(__name__)


class Scattering1DFileType(StrEnum):
    """
    Enum for different types of 1D SAXS files.
    """
    DAT = 'atsas_dat'


class Scattering1DFile(FileContainer[Scattering1DFileType, 'ScatteringProfile'],
                      abc.ABC):
    """
    Base class for reading 1D SAXS files.
    """
    file_type: Scattering1DFileType
    """Type of reflections file"""


class Scattering1DFileReaders(FileReaderMeta[Scattering1DFileType, Scattering1DFile,
                                            'ScatteringProfile']):
    """
    Metaclass for registering scattering profile file types.
    This allows for dynamic registration of new file types.
    """
    extensions = {
        Scattering1DFileType.DAT: ['.dat'],
    }
    base_class = Scattering1DFile


read_scattering_profile = Scattering1DFileReaders.read_file
"""Generic reader for diffraction reflections files."""


class AtsasDatScatteringFile(Scattering1DFile, metaclass=Scattering1DFileReaders):
    """
    Class for reading ATSAS DAT files.
    """
    file_type = Scattering1DFileType.DAT

    _METADATA_RE: re.Pattern = \
        re.compile(
            r'^(' + '|'.join(re.escape(k)
                             for k in [
                                 'Sample description',
                                 'Sample',
                                 'Parent(s)',
                                 'working-directory',
                                 'angular-axis-file',
                                 'beamstop-mask-file',
                                 'creator',
                                 'creator-version'
                             ])
            + r')\s*:\s*(.*)$',
            re.IGNORECASE
        )
    _DATA_RE: re.Pattern = re.compile(
        r'^\s*'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
    )

    @staticmethod
    def sniff(file: str | Path) -> bool:
        """
        Check if the file is a valid ATSAS DAT file by looking for key-value pairs and
        three columns of data.
        """
        file = Path(file)
        with file.open('r') as f:
            lines = f.readlines()
            metadata_lines, data_lines = 0, 0
            for line in lines:
                if metadata_lines > 0 and data_lines >= 10:
                    # 10 data points and _some_ metadata
                    return True

                line = line.rstrip('\n')
                metadata = AtsasDatScatteringFile._METADATA_RE.match(line)
                if metadata and len(metadata.groups()) == 2:
                    metadata_lines += 1

                data = AtsasDatScatteringFile._DATA_RE.match(line)
                if data and len(data.groups()) == 3:
                    data_lines += 1

        return False

    def read(self) -> 'ScatteringProfile':
        """
        Read the DAT file using gemmi.
        """
        from xtl.datasets.scattering.data import ScatteringData
        from xtl.datasets.scattering.dtypes import ANGULAR_DTYPES, INTENSITY_DTYPES, SIGMA_DTYPES
        from xtl.scattering.profiles import ScatteringProfile
        from xtl.scattering.metadata import AtsasDatScatteringProfileMetadata

        # Read file contents
        # NB: UTF-8 is required for parsing special characters (e.g. deg/A)
        with open(self.file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Separate data from metadata
        metadata_lines = []
        data_lines = []
        for line in lines:
            line = line.rstrip('\n')
            if not line:
                # Skip empty lines
                continue
            if match := self._DATA_RE.match(line):
                # First check if the line contains 3 columns of numbers
                data_lines.append(match.groups())
            else:
                # If not, assume it is metadata
                metadata_lines.append(line)

        # Sanitize metadata into key-valued pairs
        metadata = {}
        for line in metadata_lines:
            groups = line.split(':', maxsplit=1)
            if len(groups) != 2:
                continue

            key, value = groups[0].strip(), groups[1].strip()
            if key == 'Parent(s)':
                value = [v for v in value.split(' ') if v]

            metadata[key] = value

        # Convert to metadata object
        metadata = AtsasDatScatteringProfileMetadata.from_dat_kwargs(metadata)

        # Determine radial axis units
        #  Probably always q_nm, but we shall warn when it is not...
        if not metadata.angular_unit:
            logger.warning('No angular unit specified in file: %s', self.file)
        elif metadata.angular_unit != 'nanometer':
            logger.warning('Unexpected angular unit \'%s\' in file: %s', metadata.angular_unit, self.file)

        # Determine wavelength
        if not metadata.wavelength:
            logger.warning('No wavelength specified in file: %s', self.file)
            wavelength = None
        else:
            wavelength = metadata.wavelength * 10  # Convert nm -> A

        data = ScatteringData(data=data_lines, columns=('q', 'I', 'sigma'), radial_col='q', wavelength=wavelength)
        data = data.astype(
            {
                'q': ANGULAR_DTYPES['q_nm'],
                'I': INTENSITY_DTYPES['arbitrary'],
                'sigma': SIGMA_DTYPES['arbitrary']
            }
        )

        return ScatteringProfile(data=data_lines, metadata=metadata, wavelength=wavelength)
