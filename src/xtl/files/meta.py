import abc
from enum import Enum
from pathlib import Path
from typing import TypeVar, Generic, Dict, List, Set, Type

from xtl.common.compatibility import PY310_OR_LESS

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


# Type variables for generic typing
FileTypeEnum = TypeVar('FileTypeEnum', bound=StrEnum)
DataType = TypeVar('DataType')
FileContainerType = TypeVar('FileContainerType', bound='FileContainer')


class FileContainer(abc.ABC, Generic[FileTypeEnum, DataType]):
    """
    Base class for wrapping files with a specific type and providing a reader interface.
    """
    file_type: FileTypeEnum
    """Type of the file, defined by an enum"""

    def __init__(self, file: str | Path):
        """
        Initialize the file container with the given file.
        """
        self._filename = Path(file)
        if not self._filename.exists():
            raise FileNotFoundError(f'File not found: {self._filename}')

    @property
    def file(self) -> Path:
        """Path to the file."""
        return self._filename

    @staticmethod
    @abc.abstractmethod
    def sniff(file: str | Path) -> bool:
        """
        Check if the provided file is of the expected type.

        :param file: Path to the file
        :return: True if the file can be read by this container, False otherwise
        """
        pass

    @abc.abstractmethod
    def read(self) -> DataType:
        """
        Read the file and return the data.
        """
        pass


class FileReaderMeta(abc.ABCMeta, Generic[FileContainerType, FileTypeEnum, DataType]):
    """
    A generic metaclass for registering file readers that also provides a shared
    interface for reading files based on their extensions and content.

    :type FileContainerType: The type of the file reader class must be a subclass of
        FileContainer[FileTypeEnum, DataType]
    :type FileTypeEnum: Enum class defining the file types
    :type DataType: The type of data returned by the file reader
    """
    registry: Dict[FileTypeEnum, List[Type[FileContainerType]]] = {}
    """Registry of file readers, mapping file types to their corresponding classes"""

    # To be set by the subclasses
    extensions: Dict[FileTypeEnum, List[str]] = {}
    """Mapping of file types to their known file extensions"""

    base_class: Type[FileContainerType] = None
    """Base class that all file readers must inherit from."""

    def __new__(mcs, name, bases, namespace):
        """
        Create a new class with the given name, bases, and namespace.
        This method validates the class definition and registers it in the registry.

        :param name: Name of the class
        :param bases: Tuple of base classes
        :param namespace: Dictionary containing class attributes
        :raises TypeError: If the class does not meet the requirements
        :return: The newly created class
        """
        # Ensure metaclass has defined extensions and base_class
        if not mcs.extensions:
            raise TypeError(f'Metaclass {mcs.__name__} must define at least one file '
                            f'type in file types in \'extensions\'')
        if not mcs.base_class:
            raise TypeError(f'Metaclass {mcs.__name__} must define a base class in '
                            f'\'base_class\'')

        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for the base class itself
        if name == mcs.base_class.__name__ if mcs.base_class else None:
            return cls

        # Check if the class inherits from the base class
        if not issubclass(cls, mcs.base_class):
            raise TypeError(f'Class {cls.__name__} must inherit from '
                            f'{mcs.base_class.__name__}')

        # Skip abstract classes
        if abc.ABC in bases:
            return cls
        elif hasattr(cls, '__abstractmethods__') and cls.__abstractmethods__:
            return cls

        # Validate the file_type attribute
        if 'file_type' not in cls.__dict__:
            raise TypeError(f'Class {cls.__name__} must define a '
                            f'\'file_type\' attribute')

        # Get the expected type from the first key in extensions
        expected_type = next(iter(mcs.extensions.keys())).__class__
        if not isinstance(cls.file_type, expected_type):
            raise TypeError(f'Class {cls.__name__} must define a \'file_type\' '
                            f'attribute of type {expected_type.__name__}')

        # Register the class
        cls_enum = cls.file_type
        if cls_enum not in mcs.registry:
            mcs.registry[cls_enum] = []
        mcs.registry[cls_enum].append(cls)

        return cls

    @classmethod
    def _get_file_types(cls, path: str | Path) -> List[FileTypeEnum]:
        """
        Returns a list of possible file types based on the file extension.

        :param path: Path to the file
        :raises ValueError: If the file extension is not supported
        :return: A list of file type enums
        """
        path = Path(path)
        file_types = []
        supported_ext = set()
        for file_type, extensions in cls.extensions.items():
            for ext in extensions:
                supported_ext.add(ext)
                # More robust check for file extension, e.g., '.tar.gz'
                if path.name.endswith(ext):
                    file_types.append(file_type)
        if file_types:
            return file_types
        raise ValueError(f'Unsupported file extension for file: {path}.\n'
                         f'Supported extensions: {", ".join(supported_ext)}')

    @classmethod
    def read_file(cls, path: str | Path) -> DataType:
        """
        Read a file based on its type. A list of possible file types is first determined
        based on the file extension. Then, all registered readers for those file types
        are checked iteratively to find one that can read the file. The first reader
        that can read the file is used to read the data, which are then returned.

        :param path: Path to the file
        :return: Data read from the file
        :raises FileNotFoundError: If the file does not exist or is not a file
        :raises NotImplementedError: If no reader is registered for the file type
        :raises ValueError: If no registered reader can read the file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')
        if not path.is_file():
            raise FileNotFoundError(f'Path is not a file: {path}')

        # Get a list of potential readers based on the file type
        file_types = cls._get_file_types(path)
        readers: Set[Type[FileContainerType]] = set()
        for file_type in file_types:
            if file_type not in cls.registry:
                continue
            readers.update(cls.registry[file_type])

        if not readers:
            raise NotImplementedError(f'No reader registered for file type '
                                      f'{file_types}')

        # Try to read the file with each registered reader
        for reader_class in readers:
            # Skip readers that fail to sniff the file
            if not reader_class.sniff(path):
                continue
            reader: FileContainerType = reader_class(path)
            return reader.read()
        raise ValueError(f'No registered reader can read the file {path} of type '
                         f'{file_types}')

