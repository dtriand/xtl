import os.path
from dataclasses import dataclass, field
from pathlib import Path
import re


def default_fstring_subkeys():
    return {
        'raw_data_dir': ['raw_data_dir', 'dataset_dir'],
        'processed_data_dir': ['processed_data_dir', 'dataset_dir']
    }


@dataclass
class DiffractionDataset:
    dataset_name: str
    dataset_dir: str
    raw_data_dir: Path
    processed_data_dir: Path = None

    # Extra attributes to be determined during __post_init__
    #  Can be overriden by classmethods
    _first_image: str = None
    _file_ext: str = ''
    _is_compressed: bool = False
    _is_h5: bool = False

    fstring_raw_data_dir: str = "{raw_data_dir}/{dataset_dir}"
    fstring_processed_data_dir: str = "{processed_data_dir}/{dataset_dir}"
    _fstring_subkeys: dict[str, list[str]] = field(default_factory=default_fstring_subkeys)

    def __post_init__(self):
        # Check that raw_data_dir exists
        raw_data_dir = self.get_raw_data_dir()
        if not raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

        # Determine the first_image
        if self._first_image is None:
            self._first_image = self._determine_first_image()

        # Determine file extension and other flags
        if not self._file_ext:
            _, self._file_ext = self._get_file_stem_and_extension(self._first_image)
        if '.gz' in self._file_ext:
            self._is_compressed = True
        if self._file_ext == '.h5':
            self._is_h5 = True
            # TODO: Add support for .h5 files
            raise NotImplementedError("HDF5 files are not yet supported.")

    @property
    def first_image(self) -> str:
        """
        The filename of the first image, not the full path.
        """
        return self._first_image

    def check_path_fstring(self, fstring_type: str):
        """
        Check that the f-string provided for the given fstring_type is valid.
        """
        # Check that the type of fstring provided is exists
        if fstring_type not in self._fstring_subkeys.keys():
            raise ValueError(f"Invalid fstring_type: {fstring_type}. "
                             f"Available options are: {self._fstring_subkeys.keys()}")
        fstring = getattr(self, f'fstring_{fstring_type}')
        subkeys = self._fstring_subkeys[fstring_type]

        # Check that all required subkeys are present in the fstring
        for subkey in subkeys:
            if f'{{{subkey}}}' not in fstring:
                raise ValueError(f"Invalid fstring_{fstring_type}: {fstring}. Missing subkey: {subkey}")

        # Check that there are no extra subkeys in the fstring
        all_subkeys = re.findall(r'{(.*?)}', fstring)
        for subkey in all_subkeys:
            if subkey not in subkeys:
                raise ValueError(f"Invalid fstring_{fstring_type}: {fstring}. Unexpected subkey: {subkey}")

    def get_raw_data_dir(self):
        self.check_path_fstring('raw_data_dir')
        subkeys = {key: getattr(self, key) for key in self._fstring_subkeys['raw_data_dir']}
        return Path(self.fstring_raw_data_dir.format(**subkeys))

    def get_processed_data_dir(self):
        self.check_path_fstring('processed_data_dir')
        subkeys = {key: getattr(self, key) for key in self._fstring_subkeys['processed_data_dir']}
        return Path(self.fstring_processed_data_dir.format(**subkeys))

    @classmethod
    def from_image(cls, image: str | Path, raw_dataset_dir: str | Path = None, processed_data_dir: str | Path = None):
        """
        Create a DiffractionDataset object from the path to an image in the dataset. It works both with
        compressed and uncompressed images. If `raw_dataset_dir` is not provided, it will be assumed that `dataset_dir`
        is the parent directory of the image and `raw_dataset_dir` is the parent directory of `dataset_dir`,
        otherwise the `dataset_dir` will be the relative path from `raw_dataset_dir` to the `image`. If the
        `processed_data_dir` is not provided, it will be assumed to be the current directory.
        """
        # Extract file name and extension, accounting for compressed files
        image = Path(image)
        file_stem, extension = cls._get_file_stem_and_extension(image)

        # Process file extension
        is_h5 = True if '.h5' in extension else False
        is_compressed = True if '.gz' in extension else False

        # Determine dataset name
        dataset_name = cls._determine_dataset_name(filename=file_stem, is_h5=is_h5)

        # Determine dataset_dir and raw_dataset_dir
        if raw_dataset_dir is None:
            dataset_dir = image.parent.name
            raw_dataset_dir = image.parent.parent
        else:
            # dataset_dir is the relative path from raw_dataset_dir to the first_image
            raw_dataset_dir = Path(raw_dataset_dir)
            dataset_dir = Path(os.path.relpath(path=image.parent, start=raw_dataset_dir))
            dataset_dir = str(dataset_dir.as_posix())  # convert to string with forward slashes
            if dataset_dir.startswith('.'):  # dataset_dir is the same or outside raw_dataset_dir
                raise ValueError(f"Invalid 'raw_dataset_dir' provided: {raw_dataset_dir}. "
                                 f"It does not seem to contain the 'image': {image}")

        # Determine processed_data_dir
        processed_data_dir = Path(processed_data_dir) if processed_data_dir else Path('.')

        # Create and return the DiffractionDataset object
        return cls(dataset_name=dataset_name, dataset_dir=dataset_dir, raw_data_dir=raw_dataset_dir,
                   processed_data_dir=processed_data_dir, _file_ext=extension,
                   _is_compressed=is_compressed, _is_h5=is_h5)

    @staticmethod
    def _get_file_stem_and_extension(image: str | Path) -> tuple[str, str]:
        """
        Extract the file stem and extension from a file name. If the file is compressed, the extension will be the
        compound extension, e.g. '.cbf.gz'
        """
        image = Path(image)
        if image.suffix == '.gz':
            uncompressed = Path(image.name.split('.gz')[0])
            return uncompressed.stem, uncompressed.suffix + '.gz'
        return image.stem, image.suffix

    @staticmethod
    def _determine_dataset_name(filename: str, is_h5: bool) -> str:
        """
        Guess the dataset name from the filename. Starts by splitting the filename from the right side on underscores.
        If the filename is an HDF5 file, it will return the part of the filename preceding the first 'master' or 'data'
        fragment encountered, e.g. 'dataset_X_Y_Z_master.h5' -> 'dataset_X_Y_Z'.
        Otherwise, it will return the part of the filename preceding the first numeric fragment,
        e.g. 'dataset_X_Y_Z_NNNNN.cbf' -> 'dataset_X_Y_Z'.
        If no match is found, or the filename does not include any underscores, then it returns the filename as is.
        """
        segment = filename
        while True:
            # if filename cannot be further fragmented, return the last segment
            if '_' not in segment:
                return segment
            # Split the filename on the last '_'
            segment, fragment = segment.rsplit('_', 1)

            # Break condition
            if is_h5:
                # For filenames such as: 'dataset_X_Y_Z_master.h5' or 'dataset_X_Y_Z_data_NNNNN.h5'
                if fragment in ['master', 'data']:
                    return segment  # anything preceding 'master' or 'data'
            else:
                # For filename such as 'dataset_X_Y_Z_NNNNN.cbf' or 'dataset_X_Y_Z_NNNNN.cbf.gz'
                if fragment.isnumeric():
                    return segment  # anything preceding the numeric fragment

    def _glob_directory(self, directory: Path, pattern: str, files_only: bool = False) -> list[Path]:
        """
        Glob a directory for files that match the given pattern. If no matches are found, raise a FileNotFoundError.
        """
        files = sorted(directory.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No matches found in directory: {directory} with pattern: {pattern}")
        if files_only:
            files = [file for file in files if not file.is_dir()]
        if not files:
            raise FileNotFoundError(f"No files found in directory: {directory} with pattern: {pattern}")
        return files

    def _get_all_images(self) -> list[Path]:
        image_dir = self.get_raw_data_dir()
        search_pattern = f'*{self._file_ext}'
        images = self._glob_directory(directory=image_dir, pattern=search_pattern, files_only=True)
        return images

    def _get_dataset_images(self) -> list[Path]:
        image_dir = self.get_raw_data_dir()
        search_pattern = f'{self.dataset_name}*{self._file_ext}'
        images = self._glob_directory(directory=image_dir, pattern=search_pattern, files_only=True)
        return images

    def _determine_first_image(self) -> str:
        """
        Determine the first image in the dataset by searching the raw_data_dir for files that match the dataset name
        and picking the first in alphabetic order.
        """
        for image in self._get_dataset_images():
            if not image.is_dir():
                return image.name