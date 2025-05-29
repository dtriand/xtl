import pytest
from xtl.common.tables import MissingValue, MissingValueConfig, Table, XTLUndefined


class TestMissingValueConfig:
    """Tests for the MissingValueConfig class."""

    @pytest.mark.parametrize(
        'values,        repr_value, expected_values', [
        ('N/A',         '-',        {'N/A'}),
        ([None, 'N/A'], 'MISSING',  {None, 'N/A'}),
        ([],            None,       set()),
        (XTLUndefined,  None,       set()),
        ({'x', 'y'},    'MISSING',  {'x', 'y'})
    ])
    def test_initialization(self, values, repr_value, expected_values):
        """Test MissingValueConfig initialization with different values."""
        missing = MissingValueConfig(values=values, repr=repr_value)
        assert missing.values == expected_values
        assert missing.repr == repr_value
        assert missing.checker is None

    def test_contains_method(self):
        """Test __contains__ method for checking missing values."""
        missing = MissingValueConfig(values=['N/A', None], repr='-')
        assert 'N/A' in missing
        assert None in missing
        assert 'XYZ' not in missing
        assert '' not in missing

    def test_custom_checker(self):
        """Test custom checker function for determining missing values."""
        # Checker that considers empty strings and zero as missing
        def is_empty_or_zero(val):
            return val == '' or val == 0

        missing = MissingValueConfig(values=['N/A', None], repr='-',
                                     checker=is_empty_or_zero)

        assert 'N/A' in missing  # Is explicitly defined as missing
        assert None in missing
        assert '' in missing  # Should be considered missing due to checker
        assert 0 in missing
        assert 'XYZ' not in missing  # Not defined as missing
        assert 1 not in missing


class TestTable:
    """Tests for Table class."""

    @pytest.fixture
    def sample_table(self):
        """Create a sample table for testing with some missing values."""
        return Table(data=[
            [1, 'Alpha', 10.5, True, 'Red'],
            [2, 'Beta', 20.3, False, 'N/A'],
            [3, 'Gamma', 30.7, True, None],
            [4, 'Delta', 40.2, False, 'Blue'],
            [5, 'Epsilon', 50.9, True, '']
        ], headers=['ID', 'Name', 'Value', 'Active', 'Color'],
           missing_values=['N/A', None, ''],
           missing_value_repr='MISSING')


    class TestMissingValues:
        """Tests for Table class with missing values."""

        def test_table_with_missing_values(self):
            """Test creating a table with missing values."""
            table = Table(data=[
                ['A', 1, 'x'],
                ['B', 'N/A', 'y'],
                ['C', 3, None]
            ], headers=['Col1', 'Col2', 'Col3'],
               missing_values=['N/A', None],
               missing_value_repr='-')

            # Check that missing values are properly represented
            assert table.get_col('Col2')[1] == '-'  # 'N/A' should be represented as '-'
            assert table.get_col('Col3')[2] == '-'  # None should be represented as '-'

            # Non-missing values should remain unchanged
            assert table.get_col('Col1')[0] == 'A'
            assert table.get_col('Col2')[0] == 1

        def test_changing_missing_values(self):
            """Test changing missing value configuration after table creation."""
            table = Table(data=[
                ['A', 1, 'x'],
                ['B', 'N/A', 'y'],
                ['C', 3, None]
            ], headers=['Col1', 'Col2', 'Col3'],
               missing_values=['N/A'],
               missing_value_repr='-')

            # Initially only 'N/A' is considered missing
            assert table.get_col('Col2')[1] == '-'  # 'N/A' should be represented as '-'
            assert table.get_col('Col3')[2] is None  # None is not considered missing yet

            # Add None to missing values
            table.missing = ['N/A', None]

            # Now None should also be represented as '-'
            assert table.get_col('Col2')[1] == '-'  # 'N/A' should still be represented as '-'
            assert table.get_col('Col3')[2] == '-'  # None should now be represented as '-'

            # Change the representation to 'MISSING'
            table.missing = MissingValueConfig(values=['N/A', None], repr='MISSING')

            # Check that the representation has changed
            assert table.get_col('Col2')[1] == 'MISSING'
            assert table.get_col('Col3')[2] == 'MISSING'

        def test_slicing_preserves_missing_config(self):
            """Test that slicing operations preserve the missing value configuration."""
            table = Table(data=[
                ['A', 1, 'x'],
                ['B', 'N/A', 'y'],
                ['C', 3, None]
            ], headers=['Col1', 'Col2', 'Col3'],
               missing_values=['N/A', None],
               missing_value_repr='-')

            # Slice the table
            subtable = table['Col2':'Col3']

            # Check that missing values are still properly represented in the slice
            assert subtable.get_col('Col2')[1] == '-'  # 'N/A' should be represented as '-'
            assert subtable.get_col('Col3')[2] == '-'  # None should be represented as '-'

            # Change missing value representation in the original table
            table.missing = MissingValueConfig(values=['N/A', None], repr='MISSING')

            # The subtable should still have the original representation
            assert subtable.get_col('Col2')[1] == '-'
            assert subtable.get_col('Col3')[2] == '-'

            # Whereas the original table should have the new representation
            assert table.get_col('Col2')[1] == 'MISSING'
            assert table.get_col('Col3')[2] == 'MISSING'


    class TestSlicing:
        """Tests for slicing operations on the Table class."""

        def test_single_column_by_name(self, sample_table):
            """Test accessing a single column by name."""
            # Create a sliced table for testing
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table['Name']

            # Verify result is a Table instance with the correct data
            assert isinstance(result, Table)
            assert result.headers == ['Name']
            assert len(result) == 5
            assert result.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']

        def test_single_column_by_index(self, sample_table):
            """Test accessing a single column by index."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table[1]  # Should be the 'Name' column

            assert isinstance(result, Table)
            assert result.headers == ['Name']
            assert len(result) == 5
            assert result.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']

        def test_column_slice_by_indices(self, sample_table):
            """Test slicing multiple columns by indices."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table[1:3]  # Should be 'Name' and 'Value' columns

            assert isinstance(result, Table)
            assert result.headers == ['Name', 'Value']
            assert len(result) == 5
            assert result.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
            assert result.get_col('Value') == [10.5, 20.3, 30.7, 40.2, 50.9]

        def test_column_slice_by_names(self, sample_table):
            """Test slicing columns using string names."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table['ID':'Value']  # Should include 'ID', 'Name', 'Value'

            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Name', 'Value']
            assert len(result) == 5
            assert result.get_col('ID') == [1, 2, 3, 4, 5]
            assert result.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
            assert result.get_col('Value') == [10.5, 20.3, 30.7, 40.2, 50.9]

        def test_single_cell_by_name_and_index(self, sample_table):
            """Test accessing a single cell using column name and row index."""
            sliced_table = sample_table['ID':'Value', :]
            result = sliced_table['Name', 2]
            assert result == 'Gamma'  # Third row, 'Name' column

        def test_single_cell_by_indices(self, sample_table):
            """Test accessing a single cell using column and row indices."""
            sliced_table = sample_table['ID':'Value', :]
            result = sliced_table[1, 2]
            assert result == 'Gamma'  # Third row, second column ('Name')

        def test_partial_column_by_name_and_slice(self, sample_table):
            """Test slicing part of a column using column name and row slice."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table['Value', 1:4]

            assert isinstance(result, Table)
            assert result.headers == ['Value']
            assert len(result) == 3
            assert result.get_col('Value') == [20.3, 30.7, 40.2]

        def test_partial_column_by_index_and_slice(self, sample_table):
            """Test slicing part of a column using column index and row slice."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table[2, 1:4]

            assert isinstance(result, Table)
            assert result.headers == ['Value']
            assert len(result) == 3
            assert result.get_col('Value') == [20.3, 30.7, 40.2]

        def test_table_subset_by_column_and_row_slices(self, sample_table):
            """Test slicing a rectangular subset of the table."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table[1:3, 1:4]

            assert isinstance(result, Table)
            assert result.headers == ['Name', 'Value']
            assert len(result) == 3
            assert result.get_col('Name') == ['Beta', 'Gamma', 'Delta']
            assert result.get_col('Value') == [20.3, 30.7, 40.2]

        def test_single_row_slice(self, sample_table):
            """Test getting a single row as a table."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table[:, 2]

            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Name', 'Value']
            assert len(result) == 1
            assert result.get_row(0) == [3, 'Gamma', 30.7]

        def test_negative_indices(self, sample_table):
            """Test slicing with negative indices."""
            sliced_table = sample_table['ID':'Value', :]

            # Last column, second-to-last row
            result = sliced_table[-1, -2]
            assert result == 40.2  # Value column, Delta row

            # Last two columns, last two rows
            result = sliced_table[-2:, -2:]

            assert isinstance(result, Table)
            assert result.headers == ['Name', 'Value']
            assert len(result) == 2
            assert result.get_col('Name') == ['Delta', 'Epsilon']
            assert result.get_col('Value') == [40.2, 50.9]

        def test_step_in_slicing(self, sample_table):
            """Test slicing with a step value."""
            sliced_table = sample_table['ID':'Value', :]

            # Every other column
            result = sliced_table[::2]

            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Value']
            assert len(result) == 5

            # Every other row
            result = sliced_table[:, ::2]

            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Name', 'Value']
            assert len(result) == 3
            assert result.get_col('ID') == [1, 3, 5]

            # Every other column and every other row
            result = sliced_table[::2, ::2]

            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Value']
            assert len(result) == 3
            assert result.get_col('ID') == [1, 3, 5]
            assert result.get_col('Value') == [10.5, 30.7, 50.9]

        def test_empty_slice(self, sample_table):
            """Test empty slices."""
            sliced_table = sample_table['ID':'Value', :]

            result = sliced_table[2:2]  # Empty column slice
            assert isinstance(result, Table)
            assert result.headers == []
            assert len(result) == 0

            result = sliced_table[:, 2:2]  # Empty row slice
            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Name', 'Value']
            assert len(result) == 0

        def test_setting_cell_value(self, sample_table):
            """Test setting a single cell value."""
            sliced_table = sample_table['ID':'Value', :]
            sliced_table['Value', 2] = 35.5
            assert sliced_table.get_col('Value')[2] == 35.5

        def test_setting_cell_by_indices(self, sample_table):
            """Test setting a cell value using indices."""
            sliced_table = sample_table['ID':'Value', :]
            sliced_table[2, 1] = 25.0
            assert sliced_table.get_col('Value')[1] == 25.0

        def test_setting_partial_column(self, sample_table):
            """Test setting a part of a column."""
            sliced_table = sample_table['ID':'Value', :]
            sliced_table['Value', 1:3] = [25.0, 35.0]
            assert sliced_table.get_col('Value') == [10.5, 25.0, 35.0, 40.2, 50.9]

        def test_setting_entire_column(self, sample_table):
            """Test setting an entire column."""
            sliced_table = sample_table['ID':'Value', :]
            sliced_table['ID'] = [10, 20, 30, 40, 50]
            assert sliced_table.get_col('ID') == [10, 20, 30, 40, 50]

        def test_column_slice_with_string_boundaries(self, sample_table):
            """Test column slicing with string boundaries."""
            sliced_table = sample_table['ID':'Value', :]

            # From 'Name' to 'Value'
            result = sliced_table['Name':'Value']
            assert isinstance(result, Table)
            assert result.headers == ['Name', 'Value']
            assert len(result) == 5

            # From beginning to 'Name'
            result = sliced_table[:'Name']
            assert isinstance(result, Table)
            assert result.headers == ['ID', 'Name']
            assert len(result) == 5

            # From 'Value' to end
            result = sliced_table['Value':]
            assert isinstance(result, Table)
            assert result.headers == ['Value']
            assert len(result) == 5

        def test_row_slice_with_open_boundaries(self, sample_table):
            """Test row slicing with open boundaries."""
            sliced_table = sample_table['ID':'Value', :]

            # From beginning to row 3
            result = sliced_table[:, :3]
            assert isinstance(result, Table)
            assert len(result) == 3
            assert result.get_col('ID') == [1, 2, 3]

            # From row 2 to end
            result = sliced_table[:, 2:]
            assert isinstance(result, Table)
            assert len(result) == 3
            assert result.get_col('ID') == [3, 4, 5]

        def test_setting_unsupported_operations(self, sample_table):
            """Test operations that are not supported for setting values."""
            sliced_table = sample_table['ID':'Value', :]
            with pytest.raises(NotImplementedError):
                sliced_table[1:3] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

        def test_invalid_slice_indices(self, sample_table):
            """Test slicing with invalid indices."""
            sliced_table = sample_table['ID':'Value', :]

            with pytest.raises(IndexError):
                sliced_table[10]  # Column index out of range

            with pytest.raises(IndexError):
                sliced_table[:, 10]  # Row index out of range

            with pytest.raises(KeyError):
                sliced_table['NonExistent']  # Non-existent column name

            with pytest.raises(KeyError):
                sliced_table['NonExistent':'Value']  # Non-existent column name in slice


    class TestConversions:
        """Tests for Table conversion methods."""

        def test_data_property(self, sample_table):
            """Test the data property returns the correct 2D list representation."""
            # Test with regular data and missing values
            data = sample_table.data
            assert isinstance(data, list)
            assert len(data) == 5  # 5 rows
            assert len(data[0]) == 5  # 5 columns
            assert data[0] == [1, 'Alpha', 10.5, True, 'Red']
            assert data[2][1] == 'Gamma'

            # Check missing values are properly represented
            assert data[1][4] == 'MISSING'  # 'N/A' is represented as 'MISSING'
            assert data[2][4] == 'MISSING'  # None is represented as 'MISSING'
            assert data[4][4] == 'MISSING'  # Empty string is represented as 'MISSING'

        def test_to_list(self, sample_table):
            """Test the to_list method returns a flattened list of all values."""
            flat_list = sample_table.to_list()
            assert flat_list == [1, 'Alpha', 10.5, True, 'Red',
                                 2, 'Beta', 20.3, False, 'MISSING',
                                 3, 'Gamma', 30.7, True, 'MISSING',
                                 4, 'Delta', 40.2, False, 'Blue',
                                 5, 'Epsilon', 50.9, True, 'MISSING']

        def test_to_dict(self, sample_table):
            """Test the to_dict method returns a dictionary with column names as keys."""
            data_dict = sample_table.to_dict()
            assert data_dict == {
                'ID': [1, 2, 3, 4, 5],
                'Name': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                'Value': [10.5, 20.3, 30.7, 40.2, 50.9],
                'Active': [True, False, True, False, True],
                'Color': ['Red', 'MISSING', 'MISSING', 'Blue', 'MISSING']
            }

        def test_to_pandas(self, sample_table):
            """Test the to_pandas method converts to a pandas DataFrame."""
            # Skip test if pandas is not installed
            pd = pytest.importorskip('pandas')

            df = sample_table.to_pandas()
            pd.testing.assert_frame_equal(df, pd.DataFrame({
                'ID': [1, 2, 3, 4, 5],
                'Name': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                'Value': [10.5, 20.3, 30.7, 40.2, 50.9],
                'Active': [True, False, True, False, True],
                'Color': ['Red', 'MISSING', 'MISSING', 'Blue', 'MISSING']
            }))

        def test_to_numpy(self, sample_table):
            """Test the to_numpy method converts to a numpy array."""
            # Skip test if numpy is not installed
            np = pytest.importorskip('numpy')

            # Convert to numpy array
            arr = sample_table.to_numpy()
            assert (arr == np.array(
                [[1, 'Alpha', 10.5, True, 'Red'],
                 [2, 'Beta', 20.3, False, 'MISSING'],
                 [3, 'Gamma', 30.7, True, 'MISSING'],
                 [4, 'Delta', 40.2, False, 'Blue'],
                 [5, 'Epsilon', 50.9, True, 'MISSING']]
            )).all()

            # Test with a table containing only numeric data
            numeric_data = [
                [1, 10.5, 100],
                [2, 20.3, 200],
                [3, 30.7, 300]
            ]
            numeric_table = Table(data=numeric_data, headers=['A', 'B', 'C'])

            numeric_arr = numeric_table.to_numpy()
            assert (numeric_arr == np.array(numeric_data)).all()

        def test_to_rich(self, sample_table):
            """Test the to_rich method converts to a rich Table."""
            # Skip test if rich is not installed
            rich = pytest.importorskip('rich')

            # Test with regular data and missing values
            rich_table = sample_table.to_rich()
            assert isinstance(rich_table, rich.table.Table)
            assert len(rich_table.columns) == 5
            assert [col.header for col in rich_table.columns] == \
                   ['ID', 'Name', 'Value', 'Active', 'Color']

            # Test with custom cast_as function
            def cast_as_str(val):
                return f'STR:{val}'

            custom_rich_table = sample_table.to_rich(cast_as=cast_as_str)

        def test_table_without_headers(self):
            """Test conversions with a table that has no headers."""
            # Create table without headers
            table_no_headers = Table(data=[
                [1, 2, 3],
                [4, 5, 6]
            ])

            # Test data property
            assert table_no_headers.data == [[1, 2, 3], [4, 5, 6]]

            # Test to_list
            assert table_no_headers.to_list() == [1, 2, 3, 4, 5, 6]

            # Test to_dict - should use numeric indices as keys
            dict_result = table_no_headers.to_dict()
            assert len(dict_result) == 3
            assert dict_result[0] == [1, 4]
            assert dict_result[1] == [2, 5]
            assert dict_result[2] == [3, 6]

        def test_empty_table(self):
            """Test conversions with an empty table."""
            # Create empty table with headers
            empty_table = Table(headers=['A', 'B', 'C'])

            # Test data property
            assert empty_table.data == []

            # Test to_list
            assert empty_table.to_list() == []

            # Test to_dict - should return empty lists for each header
            dict_result = empty_table.to_dict()
            assert len(dict_result) == 3
            assert dict_result['A'] == []
            assert dict_result['B'] == []
            assert dict_result['C'] == []

        def test_to_csv_as_string(self, sample_table):
            """Test the to_csv method returns a valid CSV string when no filename is provided."""
            csv_str = sample_table.to_csv()

            assert csv_str == 'ID,Name,Value,Active,Color\n' \
                      '1,Alpha,10.5,True,Red\n' \
                      '2,Beta,20.3,False,MISSING\n' \
                      '3,Gamma,30.7,True,MISSING\n' \
                      '4,Delta,40.2,False,Blue\n' \
                      '5,Epsilon,50.9,True,MISSING\n'

        def test_to_csv_with_custom_params(self, sample_table):
            """Test the to_csv method with custom parameters."""
            csv_str = sample_table.to_csv(delimiter=';', header_char='#', new_line='\r\n')

            assert csv_str == '#ID;Name;Value;Active;Color\r\n' \
                      '1;Alpha;10.5;True;Red\r\n' \
                      '2;Beta;20.3;False;MISSING\r\n' \
                      '3;Gamma;30.7;True;MISSING\r\n' \
                      '4;Delta;40.2;False;Blue\r\n' \
                      '5;Epsilon;50.9;True;MISSING\r\n'

        def test_to_csv_to_file(self, sample_table, tmp_path):
            """Test the to_csv method writes to a file correctly."""
            filepath = tmp_path / 'test_table.csv'

            # Export to CSV file
            result_path = sample_table.to_csv(filename=filepath)

            # Check the returned path is correct
            assert result_path == filepath.with_suffix('.csv')

            # Check the file exists
            assert filepath.with_suffix('.csv').exists()

            # Check the file contents
            content = filepath.with_suffix('.csv').read_text()
            assert content == 'ID,Name,Value,Active,Color\n' \
                      '1,Alpha,10.5,True,Red\n' \
                      '2,Beta,20.3,False,MISSING\n' \
                      '3,Gamma,30.7,True,MISSING\n' \
                      '4,Delta,40.2,False,Blue\n' \
                      '5,Epsilon,50.9,True,MISSING\n'

            # Test overwrite parameter
            with pytest.raises(FileExistsError):
                sample_table.to_csv(filename=filepath, overwrite=False)

            # Should succeed with overwrite=True
            sample_table.to_csv(filename=filepath, overwrite=True)

            # Test keep_file_ext parameter
            ext_path = tmp_path / 'test_table.txt'
            result_path = sample_table.to_csv(filename=ext_path, keep_file_ext=True)
            assert result_path == ext_path
            assert ext_path.exists()

            # Without keep_file_ext, should add .csv extension
            ext_path = tmp_path / 'test_table2.txt'
            result_path = sample_table.to_csv(filename=ext_path, keep_file_ext=False)
            assert result_path == ext_path.with_suffix('.csv')
            assert ext_path.with_suffix('.csv').exists()


    class TestConstructors:
        """Tests for the Table class constructors."""

        def test_from_dict(self):
            """Test creating a Table from a dictionary."""
            # Create a dictionary
            data_dict = {
                'ID': [1, 2, 3, 4, 5],
                'Name': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                'Value': [10.5, 20.3, 30.7, 40.2, 50.9],
                'Active': [True, False, True, False, True]
            }

            # Create table from dict
            table = Table.from_dict(data_dict)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 5
            assert table.get_col('ID') == [1, 2, 3, 4, 5]
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
            assert table.get_col('Value') == [10.5, 20.3, 30.7, 40.2, 50.9]
            assert table.get_col('Active') == [True, False, True, False, True]

        def test_from_dict_with_missing_values(self):
            """Test creating a Table from a dictionary with missing values."""
            # Create a dictionary with missing values
            data_dict = {
                'Fruit': ['Apple', 'Banana', 'Orange', 'Grape'],
                'Quantity': [10, 'N/A', 12, 5],
                'Color': ['Red', 'Yellow', None, '']
            }

            # Create table from dict with missing values
            table = Table.from_dict(data_dict, missing_values=['N/A', None, ''],
                                    missing_value_repr='MISSING')

            # Verify the table
            assert table.headers == ['Fruit', 'Quantity', 'Color']
            assert len(table) == 4
            assert table.get_col('Fruit') == ['Apple', 'Banana', 'Orange', 'Grape']
            assert table.get_col('Quantity') == [10, 'MISSING', 12, 5]
            assert table.get_col('Color') == ['Red', 'Yellow', 'MISSING', 'MISSING']

        def test_from_dict_empty(self):
            """Test creating a Table from an empty dictionary."""
            table = Table.from_dict({})
            assert table.headers == []
            assert len(table) == 0

        def test_from_dict_unequal_lengths(self):
            """Test that creating a Table from a dictionary with unequal column lengths raises ValueError."""
            data_dict = {
                'A': [1, 2, 3],
                'B': [4, 5]  # One element short
            }
            with pytest.raises(ValueError):
                Table.from_dict(data_dict)

        def test_from_numpy(self):
            """Test creating a Table from a numpy array."""
            # Skip test if numpy is not installed
            np = pytest.importorskip('numpy')

            # Create a numpy array
            array = np.array([
                [1, 'Alpha', 10.5, True],
                [2, 'Beta', 20.3, False],
                [3, 'Gamma', 30.7, True],
                [4, 'Delta', 40.2, False],
                [5, 'Epsilon', 50.9, True]
            ])

            # Create table from numpy array with headers
            headers = ['ID', 'Name', 'Value', 'Active']
            table = Table.from_numpy(array, headers=headers)

            # Verify the table
            assert table.headers == headers
            assert len(table) == 5
            assert table.get_col('ID') == ['1', '2', '3', '4', '5']  # Note: NumPy converts to strings
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
            assert table.get_col('Value') == ['10.5', '20.3', '30.7', '40.2', '50.9']
            assert table.get_col('Active') == ['True', 'False', 'True', 'False', 'True']

        def test_from_numpy_without_headers(self):
            """Test creating a Table from a numpy array without specifying headers."""
            # Skip test if numpy is not installed
            np = pytest.importorskip('numpy')

            # Create a numpy array
            array = np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ])

            # Create table from numpy array without headers
            table = Table.from_numpy(array)

            # Verify the table
            assert table.headers == []
            assert len(table) == 3
            assert table.get_col(0) == [1, 4, 7]
            assert table.get_col(1) == [2, 5, 8]
            assert table.get_col(2) == [3, 6, 9]

        def test_from_numpy_non_2d(self):
            """Test that creating a Table from a non-2D numpy array raises ValueError."""
            # Skip test if numpy is not installed
            np = pytest.importorskip('numpy')

            # Create a 1D numpy array
            array_1d = np.array([1, 2, 3, 4, 5])

            # Create a 3D numpy array
            array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

            # Both should raise ValueError
            with pytest.raises(ValueError):
                Table.from_numpy(array_1d)

            with pytest.raises(ValueError):
                Table.from_numpy(array_3d)

        def test_from_pandas(self):
            """Test creating a Table from a pandas DataFrame."""
            # Skip test if pandas is not installed
            pd = pytest.importorskip('pandas')

            # Create a pandas DataFrame
            df = pd.DataFrame({
                'ID': [1, 2, 3, 4, 5],
                'Name': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                'Value': [10.5, 20.3, 30.7, 40.2, 50.9],
                'Active': [True, False, True, False, True]
            })

            # Create table from DataFrame
            table = Table.from_pandas(df)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 5
            assert table.get_col('ID') == [1, 2, 3, 4, 5]
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
            assert table.get_col('Value') == [10.5, 20.3, 30.7, 40.2, 50.9]
            assert table.get_col('Active') == [True, False, True, False, True]

        def test_from_pandas_with_missing_values(self):
            """Test creating a Table from a pandas DataFrame with missing values."""
            # Skip test if pandas is not installed
            pd = pytest.importorskip('pandas')
            np = pytest.importorskip('numpy')

            # Create a pandas DataFrame with missing values
            df = pd.DataFrame({
                'Fruit': ['Apple', 'Banana', 'Orange', 'Grape'],
                'Quantity': [10, np.nan, 12, 5],
                'Color': ['Red', 'Yellow', None, '']
            })

            # Create table from DataFrame with missing values
            table = Table.from_pandas(df, missing_values=[np.nan, None, ''],
                                      missing_value_repr='MISSING')

            # Verify the table
            assert table.headers == ['Fruit', 'Quantity', 'Color']
            assert len(table) == 4
            assert table.get_col('Fruit') == ['Apple', 'Banana', 'Orange', 'Grape']

            # Check the missing values are handled correctly
            quantity_col = table.get_col('Quantity')
            assert quantity_col[0] == 10
            assert quantity_col[1] == 'MISSING'
            assert quantity_col[2] == 12
            assert quantity_col[3] == 5

            assert table.get_col('Color') == ['Red', 'Yellow', 'MISSING', 'MISSING']

        def test_from_csv_string(self):
            """Test creating a Table from a CSV string."""
            csv_content = 'ID,Name,Value,Active\n' \
                          '1,Alpha,10.5,True\n' \
                          '2,Beta,20.3,False\n' \
                          '3,Gamma,30.7,True\n'

            # Create table from CSV string
            table = Table.from_csv(csv_content, header_line=0)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 3
            assert table.get_col('ID') == ['1', '2', '3']
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma']
            assert table.get_col('Value') == ['10.5', '20.3', '30.7']
            assert table.get_col('Active') == ['True', 'False', 'True']

        def test_from_csv_file(self, tmp_path):
            """Test creating a Table from a CSV file."""
            # Create a temporary CSV file
            csv_path = tmp_path / 'test.csv'
            csv_content = 'ID,Name,Value,Active\n' \
                           '1,Alpha,10.5,True\n' \
                           '2,Beta,20.3,False\n' \
                           '3,Gamma,30.7,True'
            csv_path.write_text(csv_content)

            # Create table from CSV file
            table = Table.from_csv(csv_path, header_line=0)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 3
            assert table.get_col('ID') == ['1', '2', '3']
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma']
            assert table.get_col('Value') == ['10.5', '20.3', '30.7']
            assert table.get_col('Active') == ['True', 'False', 'True']

        def test_from_csv_no_headers(self):
            """Test creating a Table from a CSV without headers."""
            csv_content = 'ID,Name,Value,Active\n' \
                          '1,Alpha,10.5,True\n' \
                          '2,Beta,20.3,False\n' \
                          '3,Gamma,30.7,True'

            # Create table from CSV string without headers
            table = Table.from_csv(csv_content)  # header_line=None by default

            # Verify the table
            assert table.headers == []
            assert len(table) == 4
            assert table.get_col(0) == ['ID', '1', '2', '3']
            assert table.get_col(1) == ['Name', 'Alpha', 'Beta', 'Gamma']
            assert table.get_col(2) == ['Value', '10.5', '20.3', '30.7']
            assert table.get_col(3) == ['Active', 'True', 'False', 'True']

        def test_from_csv_custom_delimiter(self):
            """Test creating a Table from a CSV with a custom delimiter."""
            csv_content = 'ID;Name;Value;Active\n' \
                          '1;Alpha;10.5;True\n' \
                          '2;Beta;20.3;False\n' \
                          '3;Gamma;30.7;True'

            # Create table from CSV string with semicolon delimiter
            table = Table.from_csv(csv_content, delimiter=';', header_line=0)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 3
            assert table.get_col('ID') == ['1', '2', '3']
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma']
            assert table.get_col('Value') == ['10.5', '20.3', '30.7']
            assert table.get_col('Active') == ['True', 'False', 'True']

        def test_from_csv_with_header_char(self):
            """Test creating a Table from a CSV with a header character."""
            csv_content = '# ID,Name,Value,Active\n' \
                          '1,Alpha,10.5,True\n' \
                          '2,Beta,20.3,False\n' \
                          '3,Gamma,30.7,True'

            # Create table from CSV string with # as header character
            table = Table.from_csv(csv_content, header_char='# ', header_line=0)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 3
            assert table.get_col('ID') == ['1', '2', '3']
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma']
            assert table.get_col('Value') == ['10.5', '20.3', '30.7']
            assert table.get_col('Active') == ['True', 'False', 'True']

        def test_from_csv_with_custom_newline(self):
            """Test creating a Table from a CSV with custom newline character."""
            csv_content = 'ID,Name,Value,Active\r\n' \
                          '1,Alpha,10.5,True\r\n' \
                          '2,Beta,20.3,False\r\n' \
                          '3,Gamma,30.7,True'

            # Create table from CSV string with \r\n newlines
            table = Table.from_csv(csv_content, new_line='\r\n', header_line=0)

            # Verify the table
            assert table.headers == ['ID', 'Name', 'Value', 'Active']
            assert len(table) == 3
            assert table.get_col('ID') == ['1', '2', '3']
            assert table.get_col('Name') == ['Alpha', 'Beta', 'Gamma']
            assert table.get_col('Value') == ['10.5', '20.3', '30.7']
            assert table.get_col('Active') == ['True', 'False', 'True']

        def test_from_csv_empty(self):
            """Test creating a Table from an empty CSV."""
            # Create table from empty CSV string
            table = Table.from_csv("")

            # Verify the table
            assert table.headers == []
            assert len(table) == 0

    class TestOperations:
        """Tests for table operations like adding rows and columns."""

        def test_add_row_with_list(self):
            """Test adding a new row using add_row with a list."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            # Add a new row with a list
            table.add_row([3, 'Gamma'])

            # Verify the row was added
            assert len(table) == 3
            assert table.get_row(2) == [3, 'Gamma']

        def test_add_row_with_dict(self):
            """Test adding a new row using add_row with a dictionary."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            # Add a new row with a dictionary using column names
            table.add_row({'Name': 'Gamma', 'ID': 3})

            # Verify the row was added
            assert len(table) == 3
            assert table.get_row(2) == [3, 'Gamma']

            # Add a row with a dictionary using column indices
            table.add_row({0: 4, 1: 'Delta'})

            # Verify the row was added
            assert len(table) == 4
            assert table.get_row(3) == [4, 'Delta']

        def test_add_row_with_missing_values(self):
            """Test adding a row with missing values."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name'],
                missing_values=[None],
                missing_value_repr='MISSING'
            )

            # Add a row with a missing value
            table.add_row([3, None])

            # Verify the row was added with the missing value properly represented
            assert len(table) == 3
            assert table.get_row(2) == [3, 'MISSING']

            # Add a partial row with a dictionary
            table.add_row({'ID': 4})  # No Name provided

            # Verify the row was added with missing value for Name
            assert len(table) == 4
            assert table.get_row(3) == [4, 'MISSING']

        def test_add_column_with_list(self):
            """Test adding a new column using add_col with a list."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            # Add a new column with values
            table.add_col([True, False], col_name='Active')

            # Verify the column was added
            assert table.headers == ['ID', 'Name', 'Active']
            assert table.get_col('Active') == [True, False]

        def test_add_column_with_dict(self):
            """Test adding a new column using add_col with a dictionary."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            # Add a new column with a dictionary
            table.add_col({'Active': [True, False]})

            # Verify the column was added
            assert table.headers == ['ID', 'Name', 'Active']
            assert table.get_col('Active') == [True, False]

        def test_add_empty_column(self):
            """Test adding an empty column (filled with missing values)."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name'],
                missing_values=[None],
                missing_value_repr='MISSING'
            )

            # Add an empty column
            table.add_col('Notes')

            # Verify the column was added with missing values
            assert table.headers == ['ID', 'Name', 'Notes']
            assert table.get_col('Notes') == ['MISSING', 'MISSING']

        def test_add_operator_tables(self):
            """Test the __add__ operator for combining two tables row-wise."""
            table1 = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            table2 = Table(
                data=[[3, 'Gamma'], [4, 'Delta']],
                headers=['ID', 'Name']
            )

            # Combine tables with + operator
            combined = table1 + table2

            # Verify the tables were combined correctly
            assert len(combined) == 4
            assert combined.headers == ['ID', 'Name']
            assert combined.get_col('ID') == [1, 2, 3, 4]
            assert combined.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta']

            # Ensure original tables weren't modified
            assert len(table1) == 2
            assert len(table2) == 2

        def test_iadd_operator_with_table(self):
            """Test the __iadd__ operator with another table for in-place row concatenation."""
            table1 = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            table2 = Table(
                data=[[3, 'Gamma'], [4, 'Delta']],
                headers=['ID', 'Name']
            )

            # Append table2 to table1 using +=
            table1 += table2

            # Verify the rows were appended
            assert len(table1) == 4
            assert table1.get_col('ID') == [1, 2, 3, 4]
            assert table1.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta']

        def test_iadd_operator_with_row(self):
            """Test the __iadd__ operator with a row (list or dict) for appending a row."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            # Append a row using +=
            table += [3, 'Gamma']

            # Verify the row was appended
            assert len(table) == 3
            assert table.get_row(2) == [3, 'Gamma']

            # Append a row using += with a dictionary
            table += {'ID': 4, 'Name': 'Delta'}

            # Verify the row was appended
            assert len(table) == 4
            assert table.get_row(3) == [4, 'Delta']

        def test_or_operator_tables(self):
            """Test the __or__ operator for combining two tables column-wise."""
            table1 = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            table2 = Table(
                data=[[True, 10], [False, 20]],
                headers=['Active', 'Value']
            )

            # Combine tables with | operator
            combined = table1 | table2

            # Verify the tables were combined correctly
            assert len(combined) == 2
            assert combined.headers == ['ID', 'Name', 'Active', 'Value']
            assert combined.get_row(0) == [1, 'Alpha', True, 10]
            assert combined.get_row(1) == [2, 'Beta', False, 20]

            # Ensure original tables weren't modified
            assert len(table1.headers) == 2
            assert len(table2.headers) == 2

        def test_ior_operator_with_table(self):
            """Test the __ior__ operator with another table for in-place column concatenation."""
            table1 = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            table2 = Table(
                data=[[True, 10], [False, 20]],
                headers=['Active', 'Value']
            )

            # Append columns from table2 to table1 using |=
            table1 |= table2

            # Verify the columns were appended
            assert table1.headers == ['ID', 'Name', 'Active', 'Value']
            assert table1.get_row(0) == [1, 'Alpha', True, 10]
            assert table1.get_row(1) == [2, 'Beta', False, 20]

        def test_ior_operator_with_column(self):
            """Test the __ior__ operator with a column (list, dict, or string) for adding a column."""
            table = Table(
                data=[[1, 'Alpha'], [2, 'Beta']],
                headers=['ID', 'Name']
            )

            # Add a column using |= with a dictionary
            table |= {'Active': [True, False]}

            # Verify the column was added
            assert table.headers == ['ID', 'Name', 'Active']
            assert table.get_col('Active') == [True, False]

            # Add a column using |= with a string (empty column)
            table |= 'Notes'

            # Verify the empty column was added
            assert table.headers == ['ID', 'Name', 'Active', 'Notes']
            assert len(table.get_col('Notes')) == 2

        def test_add_with_different_headers(self):
            """Test that adding tables with different headers raises an error."""
            table1 = Table(
                data=[[1, 'Alpha']],
                headers=['ID', 'Name']
            )

            table2 = Table(
                data=[[True, 10]],
                headers=['Active', 'Value']
            )

            # Attempt to add tables with different headers
            with pytest.raises(ValueError):
                combined = table1 + table2

        def test_or_with_different_row_counts(self):
            """Test that column-wise combining tables with different row counts raises an error."""
            table1 = Table(
                data=[[1, 'Alpha']],
                headers=['ID', 'Name']
            )

            table2 = Table(
                data=[[True, 10], [False, 20]],
                headers=['Active', 'Value']
            )

            # Attempt to combine tables with different row counts
            with pytest.raises(ValueError):
                combined = table1 | table2

    class TestDeletion:
        """Tests for deleting rows and columns from a Table."""

        def test_del_row(self, sample_table):
            """Test deleting a row by index."""
            # Delete the middle row (Gamma)
            sample_table.del_row(2)

            # Check that the row was deleted
            assert len(sample_table) == 4
            assert sample_table.get_col('Name') == ['Alpha', 'Beta', 'Delta', 'Epsilon']

            # Delete the first row
            sample_table.del_row(0)
            assert len(sample_table) == 3
            assert sample_table.get_col('Name') == ['Beta', 'Delta', 'Epsilon']

            # Delete the last row using negative index
            sample_table.del_row(-1)
            assert len(sample_table) == 2
            assert sample_table.get_col('Name') == ['Beta', 'Delta']

        def test_del_row_out_of_bounds(self, sample_table):
            """Test that deleting a row with an out-of-bounds index raises an error."""
            with pytest.raises(IndexError):
                sample_table.del_row(10)  # Beyond the end

            with pytest.raises(IndexError):
                sample_table.del_row(-10)  # Too far negative

        def test_del_col_by_name(self, sample_table):
            """Test deleting a column by name."""
            # Delete the 'Value' column
            sample_table.del_col('Value')

            # Check that the column was deleted
            assert sample_table.headers == ['ID', 'Name', 'Active', 'Color']
            assert sample_table.no_cols == 4

            # Check that the data was properly adjusted
            assert sample_table.get_row(0) == [1, 'Alpha', True, 'Red']
            assert sample_table.get_row(2) == [3, 'Gamma', True, 'MISSING']

        def test_del_col_by_index(self, sample_table):
            """Test deleting a column by index."""
            # Delete the second column (Name)
            sample_table.del_col(1)

            # Check that the column was deleted
            assert sample_table.headers == ['ID', 'Value', 'Active', 'Color']
            assert sample_table.no_cols == 4

            # Delete the last column using negative index
            sample_table.del_col(-1)
            assert sample_table.headers == ['ID', 'Value', 'Active']
            assert sample_table.no_cols == 3

        def test_del_col_out_of_bounds(self, sample_table):
            """Test that deleting a column with an out-of-bounds index raises an error."""
            with pytest.raises(IndexError):
                sample_table.del_col(10)  # Beyond the end

            with pytest.raises(IndexError):
                sample_table.del_col(-10)  # Too far negative

            with pytest.raises(KeyError):
                sample_table.del_col('NonExistent')  # Non-existent column name

        def test_subtraction_column_by_name(self, sample_table):
            """Test removing a column using the subtraction operator."""
            # Remove a single column by name
            result = sample_table - 'Value'

            # Check that a new table was returned with the column removed
            assert result is not sample_table  # Should be a new table
            assert result.headers == ['ID', 'Name', 'Active', 'Color']
            assert result.no_cols == 4
            assert result.no_rows == 5

            # Check that the original table is unchanged
            assert sample_table.headers == ['ID', 'Name', 'Value', 'Active', 'Color']
            assert sample_table.no_cols == 5

        def test_subtraction_multiple_columns(self, sample_table):
            """Test removing multiple columns using the subtraction operator."""
            # Remove multiple columns
            result = sample_table - ['ID', 'Active']

            # Check that a new table was returned with the columns removed
            assert result.headers == ['Name', 'Value', 'Color']
            assert result.no_cols == 3
            assert result.no_rows == 5

            # Check data integrity
            assert result.get_row(0) == ['Alpha', 10.5, 'Red']
            assert result.get_row(2) == ['Gamma', 30.7, 'MISSING']

        def test_subtraction_table_difference(self, sample_table):
            """Test computing the column-wise difference between tables."""
            # Create another table with some common columns
            other_table = Table(
                data=[[1, 'One'], [2, 'Two']],
                headers=['ID', 'Label']
            )

            # Compute the difference (columns in sample_table but not in other_table)
            result = sample_table - other_table

            # Check that only non-common columns remain
            assert 'ID' not in result.headers  # This column is in both tables
            assert 'Label' not in result.headers  # This column is only in other_table
            assert result.headers == ['Name', 'Value', 'Active', 'Color']  # These are unique to sample_table

            # Create a table with no common columns
            unrelated_table = Table(
                data=[[1, 2], [3, 4]],
                headers=['A', 'B']
            )

            # Compute the difference
            result = sample_table - unrelated_table

            # Should return the original table's columns since there are no common columns
            assert result.headers == ['ID', 'Name', 'Value', 'Active', 'Color']

            # Create a table with all common columns
            identical_headers_table = Table(
                data=[[1, 'a', 10.0, True, 'red']],
                headers=['ID', 'Name', 'Value', 'Active', 'Color']
            )

            # Compute the difference
            result = sample_table - identical_headers_table

            # Should return an empty table since all columns are common
            assert result.headers == []
            assert result.no_rows == 0

        def test_isub_column_by_name(self, sample_table):
            """Test in-place removal of a column using the -= operator."""
            # Remove a single column by name
            original = sample_table
            sample_table -= 'Value'

            # Check that the column was removed in-place
            assert sample_table is original  # Should be the same table (modified in-place)
            assert sample_table.headers == ['ID', 'Name', 'Active', 'Color']
            assert sample_table.no_cols == 4

            # Check data integrity
            assert sample_table.get_row(0) == [1, 'Alpha', True, 'Red']
            assert sample_table.get_row(2) == [3, 'Gamma', True, 'MISSING']

        def test_isub_multiple_columns(self, sample_table):
            """Test in-place removal of multiple columns using the -= operator."""
            # Remove multiple columns
            original = sample_table
            sample_table -= ['ID', 'Active']

            # Check that the columns were removed in-place
            assert sample_table is original  # Should be the same table (modified in-place)
            assert sample_table.headers == ['Name', 'Value', 'Color']
            assert sample_table.no_cols == 3

            # Check data integrity
            assert sample_table.get_row(0) == ['Alpha', 10.5, 'Red']
            assert sample_table.get_row(2) == ['Gamma', 30.7, 'MISSING']

        def test_isub_table_difference(self, sample_table):
            """Test in-place computation of the column-wise difference between tables."""
            # Create another table with some common columns
            other_table = Table(
                data=[[1, 'One'], [2, 'Two']],
                headers=['ID', 'Label']
            )

            # Compute the difference in-place
            original = sample_table
            sample_table -= other_table

            # Check that common columns were removed in-place
            assert sample_table is original  # Should be the same table (modified in-place)
            assert 'ID' not in sample_table.headers  # This column should be removed
            assert sample_table.headers == ['Name', 'Value', 'Active', 'Color']  # These should remain

            # Create a table with no common columns
            unrelated_table = Table(
                data=[[1, 2], [3, 4]],
                headers=['A', 'B']
            )

            # Current state after first subtraction
            current_headers = list(sample_table.headers)

            # Compute the difference
            sample_table -= unrelated_table

            # Should remain unchanged since there are no common columns
            assert sample_table.headers == current_headers

            # Create a table with all remaining common columns
            remaining_table = Table(
                data=[['Alpha', 10.5, True, 'Red']],
                headers=['Name', 'Value', 'Active', 'Color']
            )

            # Compute the difference
            sample_table -= remaining_table

            # Should remove all columns
            assert sample_table.headers == []
            assert sample_table.no_cols == 0
            assert sample_table.no_rows == 5  # Rows remain, just with no columns
