from Lib.io import BytesIO
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile


class NpxFile:
    _COMMENT_CHAR = '#'
    _MULTIPLIER = 2
    _NPX_TITLE = 'NUMPY EXTENDED FORMAT'
    _NPX_VERSION = '1.0'
    _NPX_HEADER_START = 'NPX_HS'
    _NPX_HEADER_END = 'NPX_HE'
    _NPX_INFO_START = 'NPX_IS'
    _NPX_INFO_END = 'NPX_IE'
    _NPX_END = 'NPX_END'

    def __init__(self, header: str = None, **kwargs):
        """
        Extended version of the Numpy ``.npy``/``.npz`` format, including additional metadata.

        :param str header: header information
        :param dict[str, np.ndarray] kwargs: datasets to save
        """
        if header is None:
            self._header = []
        else:
            self._header = header.split('\n')
        self._data = {}
        for key, array in kwargs.items():
            if isinstance(array, np.ndarray):
                self._data[key] = array

    @property
    def header(self) -> str:
        """
        File header. Includes user-stored metadata about the stored arrays.
        :return:
        """
        return '\n'.join(self._header)

    @property
    def has_header(self) -> bool:
        """
        Check if the file has an empty header.
        :return:
        """
        if len(self._header) == 0:
            return False
        return True

    @property
    def data(self) -> dict[str, np.ndarray]:
        """
        The Numpy arrays stored in the file.
        :return:
        """
        return self._data

    def _make_header(self) -> str:
        """
        Prepare the header section of the ``.npx`` file.
        :return:
        """
        header = f'{self._COMMENT_CHAR * (self._MULTIPLIER + 1)} {self._NPX_TITLE} version {self._NPX_VERSION}\n'
        header += f'{self._COMMENT_CHAR * self._MULTIPLIER} {self._NPX_HEADER_START}\n'
        header += '\n'.join(f'{self._COMMENT_CHAR} {l}' for l in self._header) + '\n'
        header += f'{self._COMMENT_CHAR * self._MULTIPLIER} {self._NPX_HEADER_END}\n'
        return header

    def _make_info(self) -> str:
        """
        Prepare the info section of the ``.npx`` file.
        :return:
        """
        info = f'{self._COMMENT_CHAR * self._MULTIPLIER} {self._NPX_INFO_START}\n'
        for name, array in self._data.items():
            info += f'{self._COMMENT_CHAR} {name}.npy: {array.shape} [{array.dtype}]\n'
        info += f'{self._COMMENT_CHAR * self._MULTIPLIER} {self._NPX_INFO_END}\n'
        return info

    def save(self, filename: str | Path, compressed: bool = True):
        """
        Export to an ``.npx`` file.

        :param str | Path filename: Output file name
        :param bool compressed: Compress the Numpy arrays
        :return:
        """
        # Prepare header
        try:
            header = self._make_header().encode(encoding='latin1')
            info = self._make_info().encode(encoding='latin1')
        except UnicodeEncodeError as e:
            print(f'Encoding error while preparing NPX header')
            print(f'Cannot encode characters \'{e.object[e.start:e.end]}\' in position {e.start}:{e.end} to encoding '
                  f'\'{e.encoding}\'')
            raise e

        # Choose filetype to save
        if self.has_header:
            file = Path(filename).with_suffix('.npx')
        else:
            file = Path(filename).with_suffix('.npz')  # save plain .npz file if there's no header information

        # Save header and data to file
        with open(file, 'wb') as f:
            if self.has_header:
                f.write(header)
                f.write(info)
                f.write(f'{self._COMMENT_CHAR * (self._MULTIPLIER + 1)} {self._NPX_END}\n'.encode(encoding='latin1'))
            if compressed:
                np.savez_compressed(f, **{name: array for name, array in self._data.items()})
            else:
                np.savez(f, **{name: array for name, array in self._data.items()})

    @classmethod
    def load(cls, filename: str | Path):
        """
        Load a ``.npx`` file.

        :param filename: Input file
        :return:
        """
        file = Path(filename)
        b = file.read_bytes()

        header_start = f'{cls._COMMENT_CHAR * cls._MULTIPLIER} {cls._NPX_HEADER_START}\n'.encode(encoding='latin1')
        header_end = f'\n{cls._COMMENT_CHAR * cls._MULTIPLIER} {cls._NPX_HEADER_END}\n'.encode(encoding='latin1')
        if b.startswith(header_start):
            # Parse a .npx file
            # Parse header
            header_block = b.split(header_start)[-1].split(header_end)[0].decode()
            header = '\n'.join([l[2:] for l in header_block.split('\n')])  # Trim comment chars

            # Parse data
            npx_end = f'{cls._COMMENT_CHAR * cls._MULTIPLIER} {cls._NPX_END}\n'.encode(encoding='latin1')
            npz_block = b.split(npx_end)[-1]
            data: NpzFile = np.load(BytesIO(npz_block))
            data_dict = {name: data[name] for name in data.files}
        elif b.startswith(b'\x93NUMPY') or b.startswith(b'PK\x03\x04') or b.startswith(b'PK\x05\x06'):
            # Parse a .npy or .npz file
            header = ''
            data: NpzFile = np.load(BytesIO(b))
            data_dict = {name: data[name] for name in data.files}
        else:
            raise Exception('Unknown file type.')
        return cls(header=header, **data_dict)


npx_load = NpxFile.load
