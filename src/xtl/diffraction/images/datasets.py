from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import os.path
import re
from typing import Optional


@dataclass
class DiffractionDataset:
    dataset_name: str
    dataset_dir: str
    raw_data_dir: Path
    processed_data_dir: Path = None
    output_dir: str = None
    fmt: str = None

    # Extra attributes to be determined during __post_init__
    #  Can be overriden by classmethods
    _first_image: Path = field(default=None, repr=False)
    _file_ext: str = field(default='', repr=False)
    _is_compressed: bool = field(default=False, repr=False)
    _is_h5: bool = field(default=False, repr=False)

    # F-strings for generating directory paths
    _fstring_dict: dict[str, str] = field(default_factory=lambda: {
        'raw_data_dir': "{raw_data_dir}/{dataset_dir}",
        'processed_data_dir': "{processed_data_dir}/{output_dir}"
    }, repr=False)
    _fstring_validator: dict[str, list[str]] = field(default_factory=lambda: {
        'raw_data_dir': ['raw_data_dir', 'dataset_dir'],
        'processed_data_dir': ['processed_data_dir', 'output_dir']
    }, repr=False)

    _dataset_images: list[Path] = field(default_factory=list, repr=False)
    _dataset_h5_images: list[Path] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # Determine the file format
        fmt = self.fmt.lower().replace('.', '') if self.fmt else None
        if fmt in ['h5', 'hdf5']:
            self._file_ext = '.h5'
            self._is_h5 = True

        # Check if a file with extension was provided instead of a dataset name
        if '.' in self.dataset_name:
            self.dataset_name = self._determine_dataset_name(filename=self.dataset_name, is_h5=self._is_h5)

        # Check that raw_data exists
        if not self.raw_data.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {self.raw_data}")

        # Determine the first_image, last_image and no_images
        self._images = self._determine_images()

        # Determine file extension and other flags
        if not self._file_ext:
            _, self._file_ext = self._get_file_stem_and_extension(self._first_image)
        if '.gz' in self._file_ext:
            self._is_compressed = True

        # Set processed_data_dir
        if self.processed_data_dir is None:
            self.processed_data_dir = Path('.')

        # Determine output_dir
        if self.output_dir is None:
            self.output_dir = self.dataset_dir

    @property
    def file_extension(self) -> str:
        """
        The file extension of the dataset images.
        """
        return self._file_ext

    @property
    def first_image(self) -> Path:
        """
        A pathlib.Path instance of the first frame in the dataset.
        """
        return self._first_image

    @property
    def last_image(self) -> Path:
        """
        A pathlib.Path instance of the last frame in the dataset.
        """
        return self._last_image

    @property
    def no_images(self) -> int:
        """
        The number of images in the dataset.
        """
        return self._no_images

    @property
    def images(self) -> list[Path]:
        """
        A list of all images in the dataset.
        """
        return self._images

    @property
    def is_h5(self) -> bool:
        """
        Whether the dataset consists of HDF5 files.
        """
        return self._is_h5

    def _check_dir_fstring(self, dir_type: str):
        """
        Check that the f-string provided for the given dir_type is valid.
        """
        # Check that the type of fstring provided is exists
        if dir_type not in self._fstring_validator.keys():
            raise ValueError(f"Invalid dir_type: {dir_type}. "
                             f"Available options are: {self._fstring_validator.keys()}")
        fstring = self._fstring_dict[dir_type]
        keys = self._fstring_validator[dir_type]

        # Check that all required keys are present in the fstring
        for key in keys:
            if f'{{{key}}}' not in fstring:
                raise ValueError(f"Invalid fstring for dir_type '{dir_type}': {fstring}. Missing key: {key}")

        # Check that there are no extra keys in the fstring
        all_keys = re.findall(r'{(.*?)}', fstring)
        for key in all_keys:
            if key not in keys:
                raise ValueError(f"Invalid fstring for dir_type '{dir_type}': {fstring}. Unexpected key: {key}")

    def register_dir_fstring(self, dir_type: str, fstring: str, keys: list[str]):
        """
        Register an f-string for programmatically generating a new directory path based on attributes of this instance.
        """
        self._fstring_dict[dir_type] = fstring
        self._fstring_validator[dir_type] = keys
        self._check_dir_fstring(dir_type)

        @property
        def custom_dir(obj):
            return obj.get_dir(dir_type)

        setattr(self.__class__, dir_type, custom_dir)

    def get_dir(self, dir_type):
        self._check_dir_fstring(dir_type)
        keys = {key: getattr(self, key) for key in self._fstring_validator[dir_type]}
        return Path(self._fstring_dict[dir_type].format(**keys))

    @property
    def raw_data(self):
        return self.get_dir('raw_data_dir')

    @property
    def processed_data(self):
        return self.get_dir('processed_data_dir')

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

            # Secondary split on the last '.' if it exists -> enables removal of file extensions
            if '.' in fragment:
                new_segment, fragment = fragment.rsplit('.', 1)
                segment = f'{segment}_{new_segment}'

            # Break condition
            if is_h5:
                # For filenames such as: 'dataset_X_Y_Z_master.h5' or 'dataset_X_Y_Z_data_NNNNN.h5'
                if fragment in ['master', 'data']:
                    return segment  # anything preceding 'master' or 'data'
            else:
                # For filename such as 'dataset_X_Y_Z_NNNNN.cbf' or 'dataset_X_Y_Z_NNNNN.cbf.gz'
                if fragment.isnumeric():
                    return segment  # anything preceding the numeric fragment

    @staticmethod
    def _glob_directory(directory: Path, pattern: str, files_only: bool = False) -> list[Path]:
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
        image_dir = self.raw_data
        search_pattern = f'*{self._file_ext}'
        images = self._glob_directory(directory=image_dir, pattern=search_pattern, files_only=True)
        return images

    def _get_dataset_images(self) -> list[Path]:
        if self._dataset_images:
            return self._dataset_images
        image_dir = self.raw_data
        search_pattern = f'{self.dataset_name}*{self._file_ext}'
        self._dataset_images = self._glob_directory(directory=image_dir, pattern=search_pattern, files_only=True)
        return self._dataset_images

    def _get_h5_master_images(self) -> list[Path]:
        image_dir = self.raw_data
        search_pattern = f'*_master{self._file_ext}'
        images = self._glob_directory(directory=image_dir, pattern=search_pattern, files_only=True)
        return images

    def _get_h5_images(self) -> list[Path]:
        if self._dataset_h5_images:
            return self._dataset_h5_images
        image_dir = self.raw_data
        search_pattern = f'{self.dataset_name}*{self._file_ext}'
        self._dataset_h5_images = self._glob_directory(directory=image_dir, pattern=search_pattern, files_only=True)
        master_image = self.first_image
        if master_image not in self._dataset_h5_images:
            raise FileNotFoundError(f"Master image {master_image} not found in directory: {image_dir}")
        self._dataset_h5_images.remove(master_image)
        self._dataset_h5_images.insert(0, master_image)
        return self._dataset_h5_images

    def reset_images_cache(self):
        """
        Reset the cache of discovered images belonging to the dataset and force execution of the glob command from
        scratch next time it is invoked. This is useful when new images might have been added in the raw_data directory.
        """
        self._dataset_images = []
        self._dataset_h5_images = []

    def get_images(self) -> list[Path]:
        """
        Returns all discovered images in the dataset, but it first resets the internal cache.
        """
        self.reset_images_cache()
        self._images = self._determine_images()
        return self._images

    def _determine_images(self) -> list[Path]:
        if self._is_h5:
            self._first_image = self._determine_h5_master_image()
            self._last_image = None
            self._no_images = None
            images = self._get_h5_images()  # run at the end, because it relies on the first_image to be set
        else:
            images = self._get_dataset_images()
            self._first_image = images[0]
            self._last_image = images[-1]
            self._no_images = len(images)
        return images

    def _determine_h5_master_image(self) -> Path:
        """
        Determine the master .h5 file in the dataset by searching the raw_data_dir for files that match
        {dataset_name}_master.h5
        """
        images = self._get_h5_master_images()
        if not images:
            raise FileNotFoundError(f"No master .h5 file found in directory: {self.raw_data}")
        for image in images:
            if image.name.startswith(self.dataset_name):
                return image
        raise FileNotFoundError(f"No master .h5 file for dataset {self.dataset_name} found in directory: "
                                f"{self.raw_data}")

    def get_all_dataset_names(self):
        """
        Get all dataset names in the raw_data_dir, if more than one dataset are present.
        """
        dataset_names = set()
        images_all = set(self._get_all_images())
        while True:
            if not images_all:
                break

            # Get an image from the set
            image = images_all.pop()

            # Determine the dataset name from the image
            dataset_name = self._determine_dataset_name(filename=image.name, is_h5=self._is_h5)
            dataset_names.add(dataset_name)

            # Grab all images in the dataset
            images_dataset = self._glob_directory(directory=self.raw_data, pattern=f'{dataset_name}*{self._file_ext}',
                                                  files_only=True)

            # Remove dataset images from all images
            images_all -= set(images_dataset)
        return sorted(list(dataset_names))

    def get_image_template(self, as_path: bool = False, first_last: bool = False) -> (
            Optional)[str | Path | tuple[str | Path, Optional[int], Optional[int]]]:
        """
        Determine a template string for the images in the dataset in the form of {dataset_name}_{####}.ext.
        This is similar to what XDS would expect. If `as_path` is True, it will return a path instance with the full
        path of the image. If `first_last` is True, an attempt will be made to determine the number of the first and
        last image during the template determination, and it will return (template, img_no_first, img_no_last).

        :param as_path: Whether to return the template as a full Path instance.
        :param first_last: Whether to return the first and last image numbers along with the template.
        """
        if self._is_h5:
            return None

        template = ''
        img_no_first, img_no_last = None, None
        # Check if the file name is in the format {dataset_name}_{####}.ext
        if '_' in self.first_image.name:
            first_image_name = self.first_image.name.replace(self.file_extension, '')
            segment, fragment = first_image_name.rsplit(sep='_', maxsplit=1)
            if segment == self.dataset_name and fragment.isnumeric():
                template = f'{self.dataset_name}_{"#" * len(fragment)}{self.file_extension}'
                # Determine the number of the first and last image
                img_no_first = int(fragment)
                last_image_name = self.last_image.name.replace(self.file_extension, '')
                img_no_last = int(last_image_name.rsplit(sep='_', maxsplit=1)[1])

        # If the above fails, try to find the longest common string between the first and last image
        if not template:
            first = self.first_image.name.replace(self.file_extension, '')
            last = self.last_image.name.replace(self.file_extension, '')

            match = SequenceMatcher(None, first, last).find_longest_match()
            if match.a == match.b == 0:
                common_template = first[match.a:match.a + match.size]
                no_digits = len(first[match.a + match.size:])
                template = f'{common_template}{"#" * no_digits}{self.file_extension}'
                try:
                    img_no_first = int(first[match.a + match.size:])
                    img_no_last = int(last[match.b + match.size:])
                except ValueError:
                    pass
        if not template:
            raise ValueError(f"Could not determine image template with first and last image: {self.first_image}, "
                             f"{self.last_image}")

        if as_path:
            template = self.raw_data / template
        if first_last:
            return template, img_no_first, img_no_last
        return template