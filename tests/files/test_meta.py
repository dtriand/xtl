import abc
import os
import pytest
from enum import auto
from pathlib import Path
from typing import List, Dict, Type

from xtl.files.meta import FileContainer, FileReaderMeta, StrEnum, FileTypeEnum, DataType, FileContainerType


# Test data setup
class TestFileType(StrEnum):
    TXT = "txt"
    CSV = "csv"
    JSON = "json"


class TextData:
    """Simple data container for text files"""
    def __init__(self, content: str):
        self.content = content


class TextFileReader(FileContainer[TestFileType, TextData], abc.ABC):
    """Abstract base class for text file readers"""


# Define metaclass for text file readers
class TextReaderMeta(FileReaderMeta[TextFileReader, TestFileType, TextData]):
    registry: Dict[TestFileType, List[Type[TextFileReader]]] = {}
    extensions = {
        TestFileType.TXT: [".txt"],
        TestFileType.CSV: [".csv"],
        TestFileType.JSON: [".json"]
    }
    base_class = TextFileReader


# Concrete implementations of text file readers
class TxtFileReader(TextFileReader, metaclass=TextReaderMeta):
    file_type = TestFileType.TXT

    @staticmethod
    def sniff(file: str | Path) -> bool:
        # Simple check: extension is .txt
        return str(file).lower().endswith(".txt")

    def read(self) -> TextData:
        with open(self.file, 'r', encoding='utf-8') as f:
            content = f.read()
        return TextData(content)


class CsvFileReader(TextFileReader, metaclass=TextReaderMeta):
    file_type = TestFileType.CSV

    @staticmethod
    def sniff(file: str | Path) -> bool:
        # Check if file has .csv extension and contains comma-separated values
        if not str(file).lower().endswith(".csv"):
            return False

        with open(file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Simple heuristic: contains at least one comma
            return "," in first_line

    def read(self) -> TextData:
        with open(self.file, 'r', encoding='utf-8') as f:
            content = f.read()
        return TextData(content)


class JsonFileReader(TextFileReader, metaclass=TextReaderMeta):
    file_type = TestFileType.JSON

    @staticmethod
    def sniff(file: str | Path) -> bool:
        # Check if file has .json extension and starts with { or [
        if not str(file).lower().endswith(".json"):
            return False

        try:
            with open(file, 'r', encoding='utf-8') as f:
                first_char = f.read(1).strip()
                return first_char in ['{', '[']
        except:
            return False

    def read(self) -> TextData:
        with open(self.file, 'r', encoding='utf-8') as f:
            content = f.read()
        return TextData(content)


# Fixtures for test files
@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory with test files"""
    # Create txt file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is a text file.")

    # Create csv file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("column1,column2,column3\nvalue1,value2,value3")

    # Create json file
    json_file = tmp_path / "test.json"
    json_file.write_text('{"key1": "value1", "key2": "value2"}')

    # Create a file with unsupported extension
    unsupported_file = tmp_path / "test.unsupported"
    unsupported_file.write_text("This file has an unsupported extension.")

    # Create an invalid json file with .json extension
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("This is not valid JSON.")

    return tmp_path


# Tests for FileContainer and subclasses
class TestFileContainer:
    def test_init_nonexistent_file(self):
        """Test that FileContainer raises FileNotFoundError for non-existent files"""
        with pytest.raises(FileNotFoundError):
            TxtFileReader("nonexistent_file.txt")

    def test_file_property(self, test_dir):
        """Test that the file property returns a Path object"""
        txt_file = test_dir / "test.txt"
        reader = TxtFileReader(txt_file)
        assert reader.file == txt_file
        assert isinstance(reader.file, Path)


# Tests for FileReaderMeta
class TestFileReaderMeta:
    def test_registry_populated(self):
        """Test that the registry is populated with the correct readers"""
        assert TestFileType.TXT in TextReaderMeta.registry
        assert TestFileType.CSV in TextReaderMeta.registry
        assert TestFileType.JSON in TextReaderMeta.registry

        # Check that the correct classes are registered
        assert TxtFileReader in TextReaderMeta.registry[TestFileType.TXT]
        assert CsvFileReader in TextReaderMeta.registry[TestFileType.CSV]
        assert JsonFileReader in TextReaderMeta.registry[TestFileType.JSON]

    def test_get_file_types(self, test_dir):
        """Test that _get_file_types returns the correct file types"""
        txt_file = test_dir / "test.txt"
        csv_file = test_dir / "test.csv"
        json_file = test_dir / "test.json"

        assert TextReaderMeta._get_file_types(txt_file) == [TestFileType.TXT]
        assert TextReaderMeta._get_file_types(csv_file) == [TestFileType.CSV]
        assert TextReaderMeta._get_file_types(json_file) == [TestFileType.JSON]

        # Test with unsupported extension
        unsupported_file = test_dir / "test.unsupported"
        with pytest.raises(ValueError):
            TextReaderMeta._get_file_types(unsupported_file)

    def test_read_file(self, test_dir):
        """Test that read_file returns the correct data"""
        txt_file = test_dir / "test.txt"
        csv_file = test_dir / "test.csv"
        json_file = test_dir / "test.json"

        # Test reading valid files
        txt_data = TextReaderMeta.read_file(txt_file)
        assert isinstance(txt_data, TextData)
        assert txt_data.content == "This is a text file."

        csv_data = TextReaderMeta.read_file(csv_file)
        assert isinstance(csv_data, TextData)
        assert "column1,column2,column3" in csv_data.content

        json_data = TextReaderMeta.read_file(json_file)
        assert isinstance(json_data, TextData)
        assert '{"key1": "value1", "key2": "value2"}' == json_data.content

        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            TextReaderMeta.read_file("nonexistent_file.txt")

        # Test reading a directory
        with pytest.raises(FileNotFoundError):
            TextReaderMeta.read_file(test_dir)

        # Test reading unsupported file
        unsupported_file = test_dir / "test.unsupported"
        with pytest.raises(ValueError):
            TextReaderMeta.read_file(unsupported_file)

        # Test reading invalid json file (should fail sniff check)
        invalid_json = test_dir / "invalid.json"
        with pytest.raises(ValueError):
            TextReaderMeta.read_file(invalid_json)

    def test_multiple_readers_for_same_type(self, test_dir):
        """Test that multiple readers can be registered for the same file type"""

        # Create a new reader for TXT files
        class AlternativeTxtReader(TextFileReader, metaclass=TextReaderMeta):
            file_type = TestFileType.TXT

            @staticmethod
            def sniff(file: str | Path) -> bool:
                # This reader only handles files with 'alternative' in the name
                return str(file).lower().endswith(".txt") and "alternative" in str(
                    file).lower()

            def read(self) -> TextData:
                with open(self.file, 'r', encoding='utf-8') as f:
                    content = f"Alternative: {f.read()}"
                return TextData(content)

        # Create an alternative txt file
        alt_txt_file = test_dir / "alternative.txt"
        alt_txt_file.write_text("This is an alternative text file.")

        # Regular txt file
        regular_txt_file = test_dir / "test.txt"

        # Verify both readers are registered
        assert len(TextReaderMeta.registry[TestFileType.TXT]) == 2
        assert TxtFileReader in TextReaderMeta.registry[TestFileType.TXT]
        assert AlternativeTxtReader in TextReaderMeta.registry[TestFileType.TXT]

        # Test sniff function
        assert TxtFileReader.sniff(regular_txt_file) is True
        assert AlternativeTxtReader.sniff(regular_txt_file) is False

        assert TxtFileReader.sniff(alt_txt_file) is True
        assert AlternativeTxtReader.sniff(alt_txt_file) is True

        # Test read_file
        # Regular file should be read by TxtFileReader (first in registry that sniffs successfully)
        regular_data = TextReaderMeta.read_file(regular_txt_file)
        assert regular_data.content == "This is a text file."

        # Alternative file should be read by AlternativeTxtReader if it comes first in the registry
        # Since registry order is not guaranteed, we'll test both readers directly
        alt_reader = AlternativeTxtReader(alt_txt_file)
        alt_data = alt_reader.read()
        assert alt_data.content == "Alternative: This is an alternative text file."

    def test_metaclass_errors(self):
        """Test that the metaclass raises appropriate errors"""

        # Test with empty extensions
        with pytest.raises(TypeError):
            class EmptyExtensionsMeta(
                FileReaderMeta[TextFileReader, TestFileType, TextData]):
                registry: Dict[TestFileType, List[Type[TextFileReader]]] = {}
                extensions = {}  # Empty extensions
                base_class = TextFileReader

            class EmptyExtensionsReader(TextFileReader, metaclass=EmptyExtensionsMeta):
                file_type = TestFileType.TXT

                def read(self) -> TextData: ...

                def sniff(self, path: str) -> bool: ...

        # Test missing base_class
        with pytest.raises(TypeError):
            class NoBaseClassMeta(
                FileReaderMeta[TextFileReader, TestFileType, TextData]):
                registry: Dict[TestFileType, List[Type[TextFileReader]]] = {}
                extensions = {
                    TestFileType.TXT: [".txt"]
                }
                # Missing base_class

            class NoBaseClassReader(TextFileReader, metaclass=NoBaseClassMeta):
                file_type = TestFileType.TXT

                def read(self) -> TextData: ...

                def sniff(self, path: str) -> bool: ...

        # Test class not inheriting from base_class
        class DifferentBase:
            pass

        class ValidMeta(FileReaderMeta[TextFileReader, TestFileType, TextData]):
            registry: Dict[TestFileType, List[Type[TextFileReader]]] = {}
            extensions = {
                TestFileType.TXT: [".txt"]
            }
            base_class = TextFileReader

        with pytest.raises(TypeError):
            class InvalidReader(DifferentBase, metaclass=ValidMeta):
                file_type = TestFileType.TXT


# Tests for concrete readers
class TestConcreteReaders:
    def test_txt_reader_sniff(self, test_dir):
        """Test that TxtFileReader.sniff correctly identifies txt files"""
        txt_file = test_dir / "test.txt"
        csv_file = test_dir / "test.csv"

        assert TxtFileReader.sniff(txt_file) is True
        assert TxtFileReader.sniff(csv_file) is False

    def test_csv_reader_sniff(self, test_dir):
        """Test that CsvFileReader.sniff correctly identifies csv files"""
        csv_file = test_dir / "test.csv"
        txt_file = test_dir / "test.txt"

        assert CsvFileReader.sniff(csv_file) is True
        assert CsvFileReader.sniff(txt_file) is False

        # Create a .csv file without commas
        no_commas_csv = test_dir / "no_commas.csv"
        no_commas_csv.write_text("This is not a CSV file")
        assert CsvFileReader.sniff(no_commas_csv) is False

    def test_json_reader_sniff(self, test_dir):
        """Test that JsonFileReader.sniff correctly identifies json files"""
        json_file = test_dir / "test.json"
        txt_file = test_dir / "test.txt"

        assert JsonFileReader.sniff(json_file) is True
        assert JsonFileReader.sniff(txt_file) is False

        # Invalid json file
        invalid_json = test_dir / "invalid.json"
        assert JsonFileReader.sniff(invalid_json) is False




