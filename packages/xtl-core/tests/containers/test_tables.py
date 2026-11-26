import locale
import math
import sys

from hypothesis import given, strategies as st
import pytest

from xtl_core.containers.tables import MissingValue, MissingValueConfig, Table

# Files are read and written in the locale's encoding, so the default encoding only shows where it isn't UTF-8
_LOCALE_IS_UTF8 = locale.getpreferredencoding(False).lower().replace('-', '') == 'utf8'


@pytest.fixture
def sample_table():
    """Five rows with missing values in the last column."""
    return Table(
        data=[
            [1, 'Alpha', 10.5, True, 'Red'],
            [2, 'Beta', 20.3, False, 'N/A'],
            [3, 'Gamma', 30.7, True, None],
            [4, 'Delta', 40.2, False, 'Blue'],
            [5, 'Epsilon', 50.9, True, ''],
        ],
        headers=['ID', 'Name', 'Value', 'Active', 'Color'],
        missing_values=['N/A', None, ''],
        missing_value_repr='MISSING',
    )


@pytest.fixture
def fmt_table():
    """Formatted columns and a missing value."""
    return Table(
        data=[[1, 'Alpha', 10.5, 1234567], [22, 'B', None, 3], [3, 'Gamma', 3.14159, 'x']],
        headers=['ID', 'Name', 'Value', 'Count'],
        missing_values=[None],
        missing_value_repr='-',
        formats={'Value': '.2f', 'Count': '>8,d'},
    )


@pytest.fixture
def small_table():
    return Table(data=[[1, 'Alpha'], [2, 'Beta']], headers=['ID', 'Name'])


class TestMissingValue:
    def test_repr(self):
        assert repr(MissingValue('N/A')) == "MissingValue('N/A')"

    def test_eq(self):
        assert MissingValue(None) == MissingValue(None)
        assert MissingValue(None) != MissingValue('')

    @pytest.mark.xfail(strict=True, reason='str() calls a to_repr() that MissingValue lacks')
    def test_str(self):
        assert str(MissingValue(1)) == '1'


class TestMissingValueConfig:
    class TestInit:
        @pytest.mark.parametrize(
            'values,        repr_value, expected_values',
            [
                ('N/A', '-', {'N/A'}),
                ([None, 'N/A'], 'MISSING', {None, 'N/A'}),
                ([], None, set()),
                ((), None, set()),
                ({'x', 'y'}, 'MISSING', {'x', 'y'}),
            ],
            ids=['single', 'list', 'empty-list', 'empty-tuple', 'set'],
        )
        def test_values(self, values, repr_value, expected_values):
            missing = MissingValueConfig(values=values, repr=repr_value)
            assert missing.values == expected_values
            assert missing.repr == repr_value
            assert missing.checker is None

        def test_repr_default(self):
            assert MissingValueConfig(values=['N/A']).repr is None

    class TestContains:
        def test_values(self):
            missing = MissingValueConfig(values=['N/A', None], repr='-')
            assert 'N/A' in missing
            assert None in missing
            assert 'XYZ' not in missing
            assert '' not in missing

        def test_checker(self):
            missing = MissingValueConfig(values=['N/A', None], repr='-', checker=lambda v: v == '' or v == 0)
            assert 'N/A' in missing
            assert None in missing
            assert '' in missing
            assert 0 in missing
            assert 'XYZ' not in missing
            assert 1 not in missing

        @pytest.mark.xfail(strict=True, reason='Unhashable values are looked up in a set')
        def test_unhashable(self):
            missing = MissingValueConfig(values=[None])
            assert [1, 2] not in missing
            assert {'a': 1} not in missing

        @pytest.mark.xfail(strict=True, reason='Unhashable values are looked up in a set')
        def test_unhashable_checker(self):
            missing = MissingValueConfig(values=[None], checker=lambda v: v == [])
            assert [] in missing

        @pytest.mark.xfail(strict=True, reason='NaN is only found when it is the same object')
        def test_nan(self):
            assert float('nan') in MissingValueConfig(values=[math.nan])

    class TestToRepr:
        def test_missing(self):
            missing = MissingValueConfig(values=['N/A'], repr='-')
            assert missing.to_repr('N/A') == '-'
            assert missing.to_repr(MissingValue('anything')) == '-'

        def test_not_missing(self):
            assert MissingValueConfig(values=['N/A'], repr='-').to_repr('x') == 'x'


class TestTable:
    class TestInit:
        def test_empty(self):
            table = Table()
            assert table.shape == (0, 0)
            assert table.headers == []

        def test_data(self):
            table = Table(data=[[1, 2], [3, 4]], headers=['a', 'b'])
            assert table.data == [[1, 2], [3, 4]]
            assert table.headers == ['a', 'b']

        def test_headers_only(self):
            assert Table(headers=['a', 'b']).shape == (0, 2)

        def test_missing_values(self):
            table = Table(
                data=[['A', 1, 'x'], ['B', 'N/A', 'y'], ['C', 3, None]],
                headers=['Col1', 'Col2', 'Col3'],
                missing_values=['N/A', None],
                missing_value_repr='-',
            )
            assert table.get_col('Col2') == [1, '-', 3]
            assert table.get_col('Col3') == ['x', 'y', '-']
            assert table.get_col('Col1') == ['A', 'B', 'C']

        def test_missing_value_config(self):
            table = Table(data=[[1, 'N/A']], missing_values=MissingValueConfig(values=['N/A'], repr='-'))
            assert table.data == [[1, '-']]

        def test_formats(self):
            assert Table(headers=['A'], formats={'A': '.2f'}).formats == {'A': '.2f'}

        def test_formats_unknown_column(self):
            with pytest.raises(KeyError):
                Table(headers=['A'], formats={'B': '.2f'})

        @pytest.mark.xfail(strict=True, reason='Unhashable values are looked up in a set')
        def test_unhashable_cells(self):
            table = Table(data=[[[1, 2], {'a': 1}]], missing_values=[None])
            assert table.data == [[[1, 2], {'a': 1}]]

        @pytest.mark.xfail(strict=True, reason='`if data:` on an array is ambiguous')
        def test_numpy_rows(self):
            np = pytest.importorskip('numpy')
            assert Table(data=np.array([[1, 2], [3, 4]])).data == [[1, 2], [3, 4]]

    class TestMissing:
        def test_get(self):
            config = MissingValueConfig(values=['N/A'], repr='-')
            assert Table(missing_values=config).missing is config

        def test_set_values_keeps_repr(self):
            table = Table(
                data=[['A', 1, 'x'], ['B', 'N/A', 'y'], ['C', 3, None]],
                headers=['Col1', 'Col2', 'Col3'],
                missing_values=['N/A'],
                missing_value_repr='-',
            )
            assert table.get_col('Col3')[2] is None

            table.missing = ['N/A', None]
            assert table.get_col('Col2')[1] == '-'
            assert table.get_col('Col3')[2] == '-'

        def test_set_config(self):
            table = Table(data=[['N/A', None]], missing_values=['N/A'], missing_value_repr='-')
            table.missing = MissingValueConfig(values=['N/A', None], repr='MISSING')
            assert table.data == [['MISSING', 'MISSING']]

        def test_set_fewer_values(self):
            table = Table(data=[['N/A', None]], missing_values=['N/A', None], missing_value_repr='-')
            table.missing = ['N/A']
            assert table.data == [['-', None]]

        def test_slices_keep_their_own(self):
            table = Table(
                data=[['A', 1, 'x'], ['B', 'N/A', 'y'], ['C', 3, None]],
                headers=['Col1', 'Col2', 'Col3'],
                missing_values=['N/A', None],
                missing_value_repr='-',
            )
            subtable = table['Col2':'Col3']
            assert subtable.get_col('Col2')[1] == '-'
            assert subtable.get_col('Col3')[2] == '-'

            table.missing = MissingValueConfig(values=['N/A', None], repr='MISSING')
            assert subtable.get_col('Col2')[1] == '-'
            assert subtable.get_col('Col3')[2] == '-'
            assert table.get_col('Col2')[1] == 'MISSING'
            assert table.get_col('Col3')[2] == 'MISSING'

    class TestSetFormat:
        def test_set_get_clear(self, fmt_table):
            fmt_table.set_format('ID', '03d')
            assert fmt_table.get_format('ID') == '03d'
            fmt_table.set_format(0, None)
            assert fmt_table.get_format('ID') is None

        def test_index_keyed_by_name(self, fmt_table):
            fmt_table.set_format(-1, ',d')
            assert fmt_table.formats['Count'] == ',d'

        def test_unknown_column(self, fmt_table):
            with pytest.raises(KeyError):
                fmt_table.set_format('Missing', '.2f')

        def test_out_of_bounds(self, fmt_table):
            with pytest.raises(IndexError):
                fmt_table.set_format(10, '.2f')

        @pytest.mark.parametrize('col', [1.0, None], ids=['float', 'none'])
        def test_column_type(self, fmt_table, col):
            with pytest.raises(TypeError):
                fmt_table.set_format(col, '.2f')

        def test_format_type(self, fmt_table):
            with pytest.raises(TypeError):
                fmt_table.set_format('ID', 2)

    class TestFormats:
        def test_copy(self, fmt_table):
            fmt_table.formats['ID'] = 'd'
            assert fmt_table.get_format('ID') is None

    class TestFormatCell:
        @pytest.mark.parametrize(
            'fmt,           value,      expected',
            [
                ('.2f', 3.14159, '3.14'),
                ('.3e', 1234.5, '1.234e+03'),
                (',d', 1234567, '1,234,567'),
                ('{:.1f} keV', 12.345, '12.3 keV'),
                ('.2f', 'abc', 'abc'),
                ('{0} {1}', 1, '1'),
            ],
            ids=['spec', 'exponent', 'thousands', 'format-string', 'wrong-type', 'invalid-format'],
        )
        def test_format(self, fmt, value, expected):
            table = Table(data=[[value]], headers=['A'], formats={'A': fmt})
            assert table._format_cell(0, table._data[0][0]) == expected

        def test_missing_not_formatted(self, fmt_table):
            assert fmt_table._format_cell(2, fmt_table._data[1][2]) == '-'

    class TestColAlign:
        @pytest.mark.parametrize(
            'fmt,           expected',
            [
                ('<.2f', '<'),
                ('*^10', '^'),
                ('=+8', '>'),
                ('{:>6.1f} A', '>'),
                ('.2f', '>'),
            ],
            ids=['left', 'fill-center', 'sign', 'format-string', 'numeric-default'],
        )
        def test_from_format(self, fmt, expected):
            table = Table(data=[[1.0], [2.0]], headers=['A'], formats={'A': fmt})
            assert table._col_align(0) == expected

        def test_defaults(self):
            table = Table(data=[[1, 'a', 'b']], headers=['A', 'B', 'C'], formats={'B': '5'})
            assert table._col_align(0) == '<'  # unformatted
            assert table._col_align(1) == '<'  # formatted, not numbers

    class TestHeaders:
        def test_get(self, small_table):
            assert small_table.headers == ['ID', 'Name']

        @pytest.mark.xfail(strict=True, reason='The internal list is returned')
        def test_get_copy(self, small_table):
            small_table.headers.append('x')
            assert small_table.headers == ['ID', 'Name']
            assert small_table.shape == (2, 2)

        def test_set(self, small_table):
            small_table.headers = ['a', 'b']
            assert small_table.get_col('a') == [1, 2]

        def test_set_wrong_length(self, small_table):
            with pytest.raises(ValueError):
                small_table.headers = ['a']

        def test_set_rekeys_formats(self, fmt_table):
            fmt_table.headers = ['a', 'b', 'c', 'd']
            assert fmt_table.formats == {'c': '.2f', 'd': '>8,d'}

        def test_set_on_table_without_headers(self):
            table = Table(data=[[1.0, 2.0], [3.0, 4.0]], formats={1: '.1e'})
            table.headers = ['A', 'B']
            assert table.formats == {'B': '.1e'}

    class TestShape:
        @pytest.mark.parametrize(
            'table, shape',
            [
                (Table(), (0, 0)),
                (Table(headers=['a', 'b']), (0, 2)),
                (Table(data=[[1, 2, 3]]), (1, 3)),
                (Table(data=[[1, 2], [3, 4]], headers=['a', 'b']), (2, 2)),
            ],
            ids=['empty', 'headers', 'no-headers', 'full'],
        )
        def test_shape(self, table, shape):
            assert table.shape == shape
            assert (table.no_rows, table.no_cols) == shape
            assert table.size == shape[0] * shape[1]
            assert len(table) == shape[0]

    class TestHasMissing:
        def test_has_missing(self, sample_table, small_table):
            assert sample_table.has_missing
            assert not small_table.has_missing

    class TestAddRow:
        def test_list(self, small_table):
            small_table.add_row([3, 'Gamma'])
            assert small_table.get_row(2) == [3, 'Gamma']

        def test_dict_names(self, small_table):
            small_table.add_row({'Name': 'Gamma', 'ID': 3})
            assert small_table.get_row(2) == [3, 'Gamma']

        def test_dict_indices(self, small_table):
            small_table.add_row({0: 4, -1: 'Delta'})
            assert small_table.get_row(2) == [4, 'Delta']

        def test_missing_values(self):
            table = Table(data=[[1, 'Alpha']], headers=['ID', 'Name'], missing_values=[None], missing_value_repr='-')
            table.add_row([3, None])
            table.add_row({'ID': 4})
            assert table.get_row(1) == [3, '-']
            assert table.get_row(2) == [4, '-']
            assert isinstance(table._data[2][1], MissingValue)

        def test_wrong_length(self, small_table):
            with pytest.raises(ValueError):
                small_table.add_row([1, 2, 3])
            assert small_table.shape == (2, 2)

        def test_unknown_name(self, small_table):
            with pytest.raises(KeyError):
                small_table.add_row({'Other': 1})
            assert small_table.no_rows == 2

        @pytest.mark.parametrize('key', [2, -3], ids=['past-end', 'before-start'])
        def test_index_out_of_range(self, small_table, key):
            with pytest.raises(IndexError):
                small_table.add_row({key: 1})
            assert small_table.no_rows == 2

        def test_first_row_sets_width(self):
            table = Table()
            table.add_row([1, 2, 3])
            assert table.shape == (1, 3)

        @pytest.mark.xfail(strict=True, reason='The length is only checked against existing rows')
        def test_wrong_length_without_rows(self):
            table = Table(headers=['a'])
            with pytest.raises(ValueError):
                table.add_row([1, 2, 3])
            assert table.shape == (0, 1)

        @pytest.mark.xfail(strict=True, reason='Empty cells are marked with a string')
        def test_dict_value_equal_to_marker(self):
            table = Table(headers=['a', 'b'], missing_value_repr='-')
            table.add_row({'a': 'XTLDummyMissingValue'})
            assert table.get_row(0) == ['XTLDummyMissingValue', '-']

        @pytest.mark.xfail(strict=True, reason='Cells are compared with the marker string by `!=`')
        def test_dict_array_value(self):
            np = pytest.importorskip('numpy')
            table = Table(headers=['a', 'b'])
            table.add_row({'a': np.array([1, 2])})
            assert table.get_row(0)[0].tolist() == [1, 2]

    class TestAddCol:
        def test_list(self, small_table):
            small_table.add_col([True, False], col_name='Active')
            assert small_table.headers == ['ID', 'Name', 'Active']
            assert small_table.get_col('Active') == [True, False]

        def test_dict(self, small_table):
            small_table.add_col({'Active': [True, False]})
            assert small_table.get_col('Active') == [True, False]

        def test_dict_several_columns(self, small_table):
            with pytest.raises(ValueError):
                small_table.add_col({'a': [1, 2], 'b': [3, 4]})

        def test_name_only(self):
            table = Table(data=[[1], [2]], headers=['ID'], missing_values=[None], missing_value_repr='-')
            table.add_col('Notes')
            assert table.headers == ['ID', 'Notes']
            assert table.get_col('Notes') == ['-', '-']

        def test_name_required_with_headers(self, small_table):
            with pytest.raises(ValueError):
                small_table.add_col([1, 2])

        def test_without_headers(self):
            table = Table(data=[[1], [2]])
            table.add_col([3, 4])
            assert table.data == [[1, 3], [2, 4]]

        def test_format(self, fmt_table):
            fmt_table.add_col([1.0, 2.0, 3.0], col_name='E', fmt='{:.1f} keV')
            assert fmt_table.get_format('E') == '{:.1f} keV'
            assert fmt_table._format_cell(4, fmt_table._data[0][4]) == '1.0 keV'

        @pytest.mark.xfail(strict=True, reason='The values are appended to rows that do not exist')
        def test_to_table_without_rows(self):
            table = Table()
            table.add_col([1, 2], col_name='a')
            assert table.headers == ['a']
            assert table.data == [[1], [2]]

        @pytest.mark.xfail(strict=True, reason='The header is added before the length is checked')
        def test_wrong_length(self, small_table):
            with pytest.raises(ValueError):
                small_table.add_col([1, 2, 3], col_name='x')
            assert small_table.headers == ['ID', 'Name']
            assert small_table.shape == (2, 2)

    class TestGetRow:
        def test_index(self, sample_table):
            assert sample_table.get_row(0) == [1, 'Alpha', 10.5, True, 'Red']
            assert sample_table.get_row(-1) == [5, 'Epsilon', 50.9, True, 'MISSING']

        def test_out_of_range(self, sample_table):
            with pytest.raises(IndexError):
                sample_table.get_row(5)

    class TestGetCol:
        def test_name(self, sample_table):
            assert sample_table.get_col('Color') == ['Red', 'MISSING', 'MISSING', 'Blue', 'MISSING']

        @pytest.mark.parametrize('idx', [1, -4], ids=['index', 'negative'])
        def test_index(self, sample_table, idx):
            assert sample_table.get_col(idx) == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']

        def test_unknown_name(self, sample_table):
            with pytest.raises(KeyError):
                sample_table.get_col('Other')

        @pytest.mark.parametrize('idx', [5, -6], ids=['past-end', 'before-start'])
        def test_out_of_range(self, sample_table, idx):
            with pytest.raises(IndexError):
                sample_table.get_col(idx)

        @pytest.mark.xfail(strict=True, reason='Without rows, every index but 0 is out of range')
        def test_without_rows(self):
            assert Table(headers=['a', 'b']).get_col(1) == []

        @pytest.mark.xfail(strict=True, reason='Without rows, index 0 is always in range')
        def test_without_columns(self):
            with pytest.raises(IndexError):
                Table().get_col(0)

    class TestSetRow:
        def test_index(self, small_table):
            small_table.set_row(1, [9, 'Nine'])
            assert small_table.get_row(1) == [9, 'Nine']

        def test_missing_values(self):
            table = Table(data=[[1, 2]], headers=['a', 'b'], missing_values=[None], missing_value_repr='-')
            table.set_row(0, [None, 3])
            assert table.get_row(0) == ['-', 3]

        def test_wrong_length(self, small_table):
            with pytest.raises(ValueError):
                small_table.set_row(0, [1])

        @pytest.mark.xfail(strict=True, reason='The length is compared with the headers')
        def test_without_headers(self):
            table = Table(data=[[1, 2]])
            table.set_row(0, [3, 4])
            assert table.data == [[3, 4]]

    class TestSetCol:
        def test_name(self, small_table):
            small_table.set_col('ID', [10, 20])
            assert small_table.get_col('ID') == [10, 20]

        def test_index(self, small_table):
            small_table.set_col(1, ['a', 'b'])
            assert small_table.get_col('Name') == ['a', 'b']

        def test_wrong_length(self, small_table):
            with pytest.raises(ValueError):
                small_table.set_col('ID', [1])

        def test_unknown_name(self, small_table):
            with pytest.raises(KeyError):
                small_table.set_col('Other', [1, 2])

        def test_out_of_range(self, small_table):
            with pytest.raises(IndexError):
                small_table.set_col(2, [1, 2])

        @pytest.mark.xfail(strict=True, reason='Negative indices are rejected')
        def test_negative_index(self, small_table):
            small_table.set_col(-1, ['a', 'b'])
            assert small_table.get_col('Name') == ['a', 'b']

    class TestData:
        def test_data(self, sample_table):
            data = sample_table.data
            assert len(data) == 5
            assert data[0] == [1, 'Alpha', 10.5, True, 'Red']
            assert [row[4] for row in data[1:3]] == ['MISSING', 'MISSING']
            assert data[4][4] == 'MISSING'

        def test_without_headers(self):
            assert Table(data=[[1, 2, 3], [4, 5, 6]]).data == [[1, 2, 3], [4, 5, 6]]

        def test_empty(self):
            assert Table(headers=['A', 'B']).data == []

    class TestToList:
        def test_flat(self, sample_table):
            flat = sample_table.to_list()
            assert len(flat) == 25
            assert flat[:5] == [1, 'Alpha', 10.5, True, 'Red']
            assert flat[9] == 'MISSING'

        def test_empty(self):
            assert Table(headers=['A']).to_list() == []

    class TestToDict:
        def test_headers(self, sample_table):
            assert sample_table.to_dict() == {
                'ID': [1, 2, 3, 4, 5],
                'Name': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                'Value': [10.5, 20.3, 30.7, 40.2, 50.9],
                'Active': [True, False, True, False, True],
                'Color': ['Red', 'MISSING', 'MISSING', 'Blue', 'MISSING'],
            }

        def test_without_headers(self):
            assert Table(data=[[1, 2, 3], [4, 5, 6]]).to_dict() == {0: [1, 4], 1: [2, 5], 2: [3, 6]}

        def test_without_rows(self):
            assert Table(headers=['A', 'B']).to_dict() == {'A': [], 'B': []}

        def test_duplicate_headers(self):
            # The first column of a name is kept
            assert Table(data=[[1, 2]], headers=['a', 'a']).to_dict() == {'a': [1]}

        @pytest.mark.xfail(strict=True, reason='The width is read from the first row')
        def test_empty(self):
            assert Table().to_dict() == {}

    class TestToNumpy:
        def test_mixed(self, sample_table):
            pytest.importorskip('numpy')
            arr = sample_table.to_numpy()
            assert arr.shape == (5, 5)
            assert arr[1, 4] == 'MISSING'

        def test_numbers(self):
            np = pytest.importorskip('numpy')
            data = [[1, 10.5, 100], [2, 20.3, 200]]
            assert (Table(data=data, headers=['A', 'B', 'C']).to_numpy() == np.array(data)).all()

    class TestToPandas:
        def test_headers(self, sample_table):
            pd = pytest.importorskip('pandas')
            pd.testing.assert_frame_equal(
                sample_table.to_pandas(),
                pd.DataFrame(
                    {
                        'ID': [1, 2, 3, 4, 5],
                        'Name': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                        'Value': [10.5, 20.3, 30.7, 40.2, 50.9],
                        'Active': [True, False, True, False, True],
                        'Color': ['Red', 'MISSING', 'MISSING', 'Blue', 'MISSING'],
                    }
                ),
            )

        def test_without_headers(self):
            pytest.importorskip('pandas')
            assert Table(data=[[1, 2]]).to_pandas().columns.tolist() == [0, 1]

    class TestToRich:
        def test_columns(self, sample_table):
            rich = pytest.importorskip('rich')
            rich_table = sample_table.to_rich()
            assert isinstance(rich_table, rich.table.Table)
            assert [col.header for col in rich_table.columns] == ['ID', 'Name', 'Value', 'Active', 'Color']

        def test_formats_and_alignment(self):
            pytest.importorskip('rich')
            table = Table(
                data=[[0, 0.0, False, None]], headers=['A', 'B', 'C', 'D'], missing_values=[None], formats={'B': '.2f'}
            )
            rich_table = table.to_rich()
            assert [col.justify for col in rich_table.columns] == ['left', 'right', 'left', 'left']
            assert [list(col.cells)[0] for col in rich_table.columns] == ['0', '0.00', 'False', '']

        def test_cast_as(self, small_table):
            pytest.importorskip('rich')
            rich_table = small_table.to_rich(cast_as=lambda v: f'<{v}>')
            assert list(rich_table.columns[0].cells) == ['<1>', '<2>']

    class TestToCsv:
        _SAMPLE_CSV = (
            'ID,Name,Value,Active,Color\n'
            '1,Alpha,10.5,True,Red\n'
            '2,Beta,20.3,False,MISSING\n'
            '3,Gamma,30.7,True,MISSING\n'
            '4,Delta,40.2,False,Blue\n'
            '5,Epsilon,50.9,True,MISSING\n'
        )

        def test_string(self, sample_table):
            assert sample_table.to_csv() == self._SAMPLE_CSV

        def test_options(self, sample_table):
            assert sample_table.to_csv(delimiter=';', header_char='#', new_line='\r\n') == (
                self._SAMPLE_CSV.replace(',', ';').replace('\n', '\r\n').replace('ID', '#ID', 1)
            )

        def test_without_headers(self):
            assert Table(data=[[1, 2], [3, 4]]).to_csv() == '1,2\n3,4\n'

        def test_formatted(self, fmt_table):
            assert fmt_table.to_csv(delimiter=';') == (
                'ID;Name;Value;Count\n1;Alpha;10.50;1,234,567\n22;B;-;3\n3;Gamma;3.14;x\n'
            )

        def test_not_formatted(self, fmt_table):
            assert fmt_table.to_csv(delimiter=';', formatted=False) == (
                'ID;Name;Value;Count\n1;Alpha;10.5;1234567\n22;B;-;3\n3;Gamma;3.14159;x\n'
            )

        def test_file(self, sample_table, tmp_path):
            path = sample_table.to_csv(filename=tmp_path / 'table.csv')
            assert path == tmp_path / 'table.csv'
            assert path.read_text() == self._SAMPLE_CSV

        def test_file_parents(self, sample_table, tmp_path):
            path = sample_table.to_csv(filename=tmp_path / 'a' / 'b' / 'table.csv')
            assert path.is_file()

        def test_file_exists(self, sample_table, tmp_path):
            path = tmp_path / 'table.csv'
            path.write_text('precious')
            with pytest.raises(FileExistsError):
                sample_table.to_csv(filename=path)
            assert path.read_text() == 'precious'

        def test_file_overwrite(self, sample_table, tmp_path):
            path = tmp_path / 'table.csv'
            path.write_text('old')
            sample_table.to_csv(filename=path, overwrite=True)
            assert path.read_text() == self._SAMPLE_CSV

        @pytest.mark.parametrize('name', ['table.txt', 'table'], ids=['other', 'none'])
        def test_file_suffix(self, sample_table, tmp_path, name):
            assert sample_table.to_csv(filename=tmp_path / name) == tmp_path / 'table.csv'

        def test_file_keep_suffix(self, sample_table, tmp_path):
            assert sample_table.to_csv(filename=tmp_path / 'table.txt', keep_file_ext=True) == tmp_path / 'table.txt'

        @pytest.mark.xfail(strict=True, reason='Only the given name is checked, then the .csv file is written')
        def test_file_suffix_exists(self, sample_table, tmp_path):
            (tmp_path / 'table.csv').write_text('precious')
            with pytest.raises(FileExistsError):
                sample_table.to_csv(filename=tmp_path / 'table.txt')
            assert (tmp_path / 'table.csv').read_text() == 'precious'

        @pytest.mark.xfail(strict=True, reason='Values are not quoted')
        def test_quoting(self):
            table = Table(data=[['a,b', 'say "hi"', 'x\ny']], headers=['h,1', 'h2', 'h3'])
            assert table.to_csv() == '"h,1",h2,h3\n"a,b","say ""hi""","x\ny"\n'

        @pytest.mark.xfail(strict=True, reason='The trailing delimiter is cut from rows without cells too')
        def test_rows_without_cells(self):
            assert Table(data=[[], []]).to_csv() == '\n\n'

        @pytest.mark.xfail(strict=True, reason='Values are stripped in unformatted columns too')
        def test_formatted_keeps_whitespace(self):
            table = Table(data=[[' a ', 1.0]], headers=['A', 'B'], formats={'B': '>6.1f'})
            assert table.to_csv() == 'A,B\n a ,1.0\n'

        @pytest.mark.xfail(strict=True, reason='No encoding parameter')
        def test_file_encoding(self, tmp_path):
            path = Table(data=[['é']]).to_csv(filename=tmp_path / 'table.csv', encoding='utf-16')
            assert path.read_text(encoding='utf-16') == 'é\n'

        @pytest.mark.xfail(not _LOCALE_IS_UTF8, strict=True, reason="Files are written in the locale's encoding")
        def test_file_utf8(self, tmp_path):
            path = Table(data=[['é', 'λ']]).to_csv(filename=tmp_path / 'table.csv')
            assert path.read_bytes().decode('utf-8') == 'é,λ\n'

        @given(
            headers=st.lists(st.text(alphabet=st.characters(exclude_characters='\x00')), min_size=1, max_size=4),
            data=st.data(),
            delimiter=st.sampled_from([',', ';', '\t', '|']),
            header_char=st.sampled_from(['', '#', '# ']),
            new_line=st.sampled_from(['\n', '\r\n']),
        )
        @pytest.mark.xfail(strict=True, reason='Values are not quoted and quotes are not parsed')
        def test_round_trip(self, headers, data, delimiter, header_char, new_line):
            # NUL is excluded: Python 3.10's csv reader rejects it
            cell = st.text(alphabet=st.characters(exclude_characters='\x00'))
            rows = data.draw(st.lists(st.lists(cell, min_size=len(headers), max_size=len(headers)), max_size=5))
            text = Table(data=rows, headers=headers).to_csv(
                delimiter=delimiter, header_char=header_char, new_line=new_line
            )
            table = Table.from_csv(text, delimiter=delimiter, header_char=header_char, new_line=new_line, header_line=0)
            assert table.headers == headers
            assert table.data == rows

    class TestFromDict:
        def test_columns(self):
            table = Table.from_dict({'ID': [1, 2], 'Name': ['Alpha', 'Beta']})
            assert table.headers == ['ID', 'Name']
            assert table.data == [[1, 'Alpha'], [2, 'Beta']]

        def test_missing_values(self):
            table = Table.from_dict(
                {'Quantity': [10, 'N/A'], 'Color': [None, '']},
                missing_values=['N/A', None, ''],
                missing_value_repr='MISSING',
            )
            assert table.data == [[10, 'MISSING'], ['MISSING', 'MISSING']]

        def test_formats(self):
            assert str(Table.from_dict({'A': [1.234]}, formats={'A': '.1f'})) == '  A\n---\n1.2'

        def test_empty(self):
            assert Table.from_dict({}).shape == (0, 0)

        def test_unequal_lengths(self):
            with pytest.raises(ValueError):
                Table.from_dict({'A': [1, 2, 3], 'B': [4, 5]})

    class TestFromNumpy:
        def test_headers(self):
            np = pytest.importorskip('numpy')
            table = Table.from_numpy(np.array([[1, 'Alpha'], [2, 'Beta']]), headers=['ID', 'Name'])
            assert table.headers == ['ID', 'Name']
            assert table.get_col('ID') == ['1', '2']  # numpy makes them all strings

        def test_without_headers(self):
            np = pytest.importorskip('numpy')
            table = Table.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
            assert table.headers == []
            assert table.get_col(2) == [3, 6]

        @pytest.mark.parametrize('shape', [(5,), (2, 2, 2)], ids=['1d', '3d'])
        def test_not_2d(self, shape):
            np = pytest.importorskip('numpy')
            with pytest.raises(ValueError):
                Table.from_numpy(np.zeros(shape))

        def test_missing_values(self):
            np = pytest.importorskip('numpy')
            table = Table.from_numpy(np.array([[1.0, -1.0]]), missing_values=[-1.0], missing_value_repr='-')
            assert table.data == [[1.0, '-']]

        @pytest.mark.xfail(strict=True, reason='NaN is checked by identity with np.nan, after tolist()')
        def test_nan(self):
            np = pytest.importorskip('numpy')
            assert Table.from_numpy(np.array([[1.0, np.nan]]), missing_value_repr='-').data == [[1.0, '-']]

        @pytest.mark.xfail(strict=True, reason='A MissingValueConfig is taken as a missing value')
        def test_missing_value_config(self):
            np = pytest.importorskip('numpy')
            config = MissingValueConfig(values=[2.0], repr='-')
            assert Table.from_numpy(np.array([[1.0, 2.0]]), missing_values=config).data == [[1.0, '-']]

    class TestFromPandas:
        def test_columns(self):
            pd = pytest.importorskip('pandas')
            table = Table.from_pandas(pd.DataFrame({'ID': [1, 2], 'Name': ['Alpha', 'Beta']}))
            assert table.headers == ['ID', 'Name']
            assert table.data == [[1, 'Alpha'], [2, 'Beta']]

        def test_missing_values(self):
            pd = pytest.importorskip('pandas')
            np = pytest.importorskip('numpy')
            df = pd.DataFrame({'Quantity': [10, np.nan, 12], 'Color': ['Red', None, '']})
            table = Table.from_pandas(df, missing_values=[''], missing_value_repr='MISSING')
            assert table.get_col('Quantity') == [10, 'MISSING', 12]
            assert table.get_col('Color') == ['Red', 'MISSING', 'MISSING']

        @pytest.mark.xfail(strict=True, reason='A MissingValueConfig is taken as a missing value')
        def test_missing_value_config(self):
            pd = pytest.importorskip('pandas')
            config = MissingValueConfig(values=['N/A'], repr='-')
            table = Table.from_pandas(pd.DataFrame({'A': ['x', 'N/A']}), missing_values=config)
            assert table.get_col('A') == ['x', '-']

    class TestFromCsv:
        _CSV = 'ID,Name,Value\n1,Alpha,10.5\n2,Beta,20.3\n'

        def test_string(self):
            table = Table.from_csv(self._CSV, header_line=0)
            assert table.headers == ['ID', 'Name', 'Value']
            assert table.data == [['1', 'Alpha', '10.5'], ['2', 'Beta', '20.3']]

        @pytest.mark.parametrize('as_path', [True, False], ids=['path', 'str'])
        def test_file(self, tmp_path, as_path):
            path = tmp_path / 'test.csv'
            path.write_text(self._CSV.rstrip())
            table = Table.from_csv(path if as_path else str(path), header_line=0)
            assert table.headers == ['ID', 'Name', 'Value']
            assert table.no_rows == 2

        def test_without_headers(self):
            table = Table.from_csv(self._CSV)
            assert table.headers == []
            assert table.get_col(0) == ['ID', '1', '2']

        def test_header_line(self):
            table = Table.from_csv('title\nID,Name\n1,Alpha\n', header_line=1)
            assert table.headers == ['ID', 'Name']
            assert table.data == [['1', 'Alpha']]

        def test_delimiter(self):
            table = Table.from_csv(self._CSV.replace(',', ';'), delimiter=';', header_line=0)
            assert table.headers == ['ID', 'Name', 'Value']

        def test_header_char(self):
            table = Table.from_csv('# ' + self._CSV, header_char='# ', header_line=0)
            assert table.headers == ['ID', 'Name', 'Value']

        def test_new_line(self):
            table = Table.from_csv(self._CSV.replace('\n', '\r\n'), new_line='\r\n', header_line=0)
            assert table.data == [['1', 'Alpha', '10.5'], ['2', 'Beta', '20.3']]

        def test_skips_empty_lines(self):
            assert Table.from_csv('1,2\n\n3,4\n').data == [['1', '2'], ['3', '4']]

        def test_formats(self):
            assert Table.from_csv(self._CSV, header_line=0, formats={'Value': '>8'}).formats == {'Value': '>8'}

        def test_empty(self):
            assert Table.from_csv('').shape == (0, 0)

        @pytest.mark.xfail(strict=True, reason='Quotes are not parsed')
        def test_quoted(self):
            table = Table.from_csv('"h,1",h2\n"a,b","say ""hi"""\n"x\ny",z\n', header_line=0)
            assert table.headers == ['h,1', 'h2']
            assert table.data == [['a,b', 'say "hi"'], ['x\ny', 'z']]

        @pytest.mark.xfail(
            sys.platform != 'win32' and sys.version_info < (3, 14),
            strict=True,
            reason='Path.exists() raises for names that are too long',
        )
        def test_long_line(self):
            assert Table.from_csv(','.join(['a'] * 300)).shape == (1, 300)

        @pytest.mark.xfail(strict=True, reason='No encoding parameter')
        def test_file_encoding(self, tmp_path):
            path = tmp_path / 'test.csv'
            path.write_text('é\n', encoding='utf-16')
            assert Table.from_csv(path, encoding='utf-16').data == [['é']]

        @pytest.mark.xfail(not _LOCALE_IS_UTF8, strict=True, reason="Files are read in the locale's encoding")
        def test_file_utf8(self, tmp_path):
            path = tmp_path / 'test.csv'
            path.write_bytes('é,λ\n'.encode('utf-8'))
            assert Table.from_csv(path).data == [['é', 'λ']]

    class TestStr:
        def test_empty(self):
            assert str(Table(headers=['A'])) == '(Empty table)'

        def test_without_headers(self):
            table = Table(data=[[1.0, 2.0], [3.0, 4.0]], formats={1: '.1e'})
            assert str(table) == '1.0 | 2.0e+00\n3.0 | 4.0e+00'

        def test_formats_and_alignment(self, fmt_table):
            assert str(fmt_table) == '\n'.join(
                [
                    'ID | Name  | Value |     Count',
                    '---+-------+-------+----------',
                    '1  | Alpha | 10.50 | 1,234,567',
                    '22 | B     |     - |         3',
                    '3  | Gamma |  3.14 |         x',
                ]
            )

    class TestGetItem:
        def test_column_by_name(self, sample_table):
            result = sample_table['Name']
            assert isinstance(result, Table)
            assert result.headers == ['Name']
            assert result.get_col('Name') == ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']

        @pytest.mark.parametrize('key', [1, -4], ids=['index', 'negative'])
        def test_column_by_index(self, sample_table, key):
            assert sample_table[key].headers == ['Name']

        def test_column_slice_by_indices(self, sample_table):
            result = sample_table[1:3]
            assert result.headers == ['Name', 'Value']
            assert result.get_col('Value') == [10.5, 20.3, 30.7, 40.2, 50.9]

        def test_column_slice_by_names(self, sample_table):
            # Both names are included
            assert sample_table['ID':'Value'].headers == ['ID', 'Name', 'Value']
            assert sample_table[:'Name'].headers == ['ID', 'Name']
            assert sample_table['Active':].headers == ['Active', 'Color']

        def test_column_slice_step(self, sample_table):
            assert sample_table[::2].headers == ['ID', 'Value', 'Color']

        def test_column_slice_negative(self, sample_table):
            assert sample_table[-2:].headers == ['Active', 'Color']

        def test_column_slice_empty(self, sample_table):
            result = sample_table[2:2]
            assert result.headers == []
            assert len(result) == 0

        @pytest.mark.parametrize('key', [(1, 2), ('Name', 2), (-4, -3)], ids=['indices', 'name', 'negative'])
        def test_cell(self, sample_table, key):
            assert sample_table[key] == 'Gamma'

        def test_cell_missing_repr(self, fmt_table):
            assert fmt_table['Value', 1] == '-'

        @pytest.mark.parametrize('key', [('Value', slice(1, 4)), (2, slice(1, 4))], ids=['name', 'index'])
        def test_column_rows(self, sample_table, key):
            result = sample_table[key]
            assert result.headers == ['Value']
            assert result.get_col('Value') == [20.3, 30.7, 40.2]

        def test_slices(self, sample_table):
            result = sample_table[1:3, 1:4]
            assert result.headers == ['Name', 'Value']
            assert result.get_col('Name') == ['Beta', 'Gamma', 'Delta']

        def test_slices_negative(self, sample_table):
            result = sample_table[-2:, -2:]
            assert result.headers == ['Active', 'Color']
            assert result.data == [[False, 'Blue'], [True, 'MISSING']]

        def test_slices_step(self, sample_table):
            result = sample_table[::2, ::2]
            assert result.headers == ['ID', 'Value', 'Color']
            assert result.get_col('ID') == [1, 3, 5]

        def test_row(self, sample_table):
            result = sample_table[:, 2]
            assert result.no_rows == 1
            assert result.get_row(0) == [3, 'Gamma', 30.7, True, 'MISSING']

        def test_row_slice_open(self, sample_table):
            assert sample_table[:, :3].get_col('ID') == [1, 2, 3]
            assert sample_table[:, 2:].get_col('ID') == [3, 4, 5]

        def test_row_slice_empty(self, sample_table):
            result = sample_table[:, 2:2]
            assert result.headers == ['ID', 'Name', 'Value', 'Active', 'Color']
            assert len(result) == 0

        @pytest.mark.parametrize(
            'key, error',
            [
                (10, IndexError),
                ((slice(None), 10), IndexError),
                ((0, 10), IndexError),
                ('Other', KeyError),
                (slice('Other', 'Value'), KeyError),
                (slice(0, 10), IndexError),
                ((0, 'a'), TypeError),
                (1.0, TypeError),
            ],
            ids=['column', 'row', 'cell-row', 'name', 'slice-name', 'slice', 'row-type', 'key-type'],
        )
        def test_invalid(self, sample_table, key, error):
            with pytest.raises(error):
                sample_table[key]

        def test_formats(self, fmt_table):
            assert fmt_table['Value':'Count'].formats == {'Value': '.2f', 'Count': '>8,d'}
            assert fmt_table['Value'].formats == {'Value': '.2f'}
            assert fmt_table[2, 0:2].formats == {'Value': '.2f'}
            assert fmt_table['ID':'Name', 0:2].formats == {}

        @pytest.mark.parametrize(
            'key',
            [
                'Value',
                slice('Name', 'Value'),
                ('Value', slice(0, 3)),
                (slice('Name', 'Value'), slice(0, 3)),
                (slice('Name', 'Value'), 1),
            ],
            ids=['column', 'columns', 'column-rows', 'slices', 'row'],
        )
        def test_missing_values(self, fmt_table, key):
            sliced = fmt_table[key]
            assert sliced.has_missing
            assert isinstance(sliced._data[1 if sliced.no_rows > 1 else 0][-1], MissingValue)

            # The slice's own representation applies
            sliced.missing = MissingValueConfig(values=[None], repr='N/A')
            assert 'N/A' in sliced.get_col('Value')

        def test_copies_rows(self, small_table):
            sliced = small_table[:]
            sliced['ID', 0] = 99
            assert small_table.get_col('ID') == [1, 2]

        @pytest.mark.xfail(strict=True, reason='The headers of the new table are looked up by index')
        def test_column_without_headers(self):
            result = Table(data=[[1, 2], [3, 4]])[1]
            assert result.headers == []
            assert result.data == [[2], [4]]

        @pytest.mark.xfail(strict=True, reason='Column slices are bounded by the number of headers')
        def test_column_slice_without_headers(self):
            assert Table(data=[[1, 2, 3]])[0:2].data == [[1, 2]]

        @pytest.mark.xfail(strict=True, reason='Column slices are bounded by the number of headers')
        def test_row_without_headers(self):
            assert Table(data=[[1, 2], [3, 4]])[:, 1].data == [[3, 4]]

        @pytest.mark.xfail(strict=True, reason='The headers of the new table are looked up by index')
        def test_cell_without_headers(self):
            assert Table(data=[[1, 2], [3, 4]])[1, 0] == 2

    class TestSetItem:
        @pytest.mark.parametrize('key', [('Value', 2), (2, 2), (2, -3)], ids=['name', 'index', 'negative-row'])
        def test_cell(self, sample_table, key):
            sample_table[key] = 35.5
            assert sample_table.get_col('Value')[2] == 35.5

        def test_column_rows(self, sample_table):
            sample_table['Value', 1:3] = [25.0, 35.0]
            assert sample_table.get_col('Value') == [10.5, 25.0, 35.0, 40.2, 50.9]

        def test_column_rows_scalar(self, sample_table):
            sample_table['Value', 1:3] = 0.0
            assert sample_table.get_col('Value') == [10.5, 0.0, 0.0, 40.2, 50.9]

        def test_column_rows_wrong_length(self, sample_table):
            with pytest.raises(ValueError):
                sample_table['Value', 1:3] = [1.0]

        def test_column(self, sample_table):
            sample_table['ID'] = [10, 20, 30, 40, 50]
            assert sample_table.get_col('ID') == [10, 20, 30, 40, 50]

        def test_missing_value(self, sample_table):
            sample_table['Name', 0] = None
            assert sample_table.get_col('Name')[0] == 'MISSING'

        def test_keeps_other_missing_values(self, fmt_table):
            fmt_table['Value', 0] = 1.0
            assert isinstance(fmt_table._data[1][2], MissingValue)
            fmt_table['Value', 0:1] = [2.0]
            assert isinstance(fmt_table._data[1][2], MissingValue)
            assert fmt_table.get_col('Value') == [2.0, '-', 3.14159]

        @pytest.mark.parametrize('key', [slice(1, 3), (slice(1, 3), 0)], ids=['columns', 'columns-row'])
        def test_column_slice_unsupported(self, sample_table, key):
            with pytest.raises(NotImplementedError):
                sample_table[key] = [1, 2]

        @pytest.mark.parametrize(
            'key, error',
            [
                (('Other', 0), KeyError),
                ((0, 10), IndexError),
                ((0, 'a'), TypeError),
                (1.0, TypeError),
            ],
            ids=['name', 'row', 'row-type', 'key-type'],
        )
        def test_invalid(self, sample_table, key, error):
            with pytest.raises(error):
                sample_table[key] = 1

        @pytest.mark.xfail(strict=True, reason='Negative column indices are rejected')
        def test_negative_column(self, sample_table):
            sample_table[-1, 0] = 'Green'
            assert sample_table.get_col('Color')[0] == 'Green'

        @pytest.mark.xfail(strict=True, reason='A string counts as a sequence of values')
        def test_column_rows_string(self, sample_table):
            sample_table['Name', 0:3] = 'xyz'
            assert sample_table.get_col('Name')[:3] == ['xyz', 'xyz', 'xyz']

    class TestIter:
        def test_rows(self, sample_table):
            rows = list(sample_table)
            assert len(rows) == 5
            assert rows[1] == [2, 'Beta', 20.3, False, 'MISSING']

    class TestItems:
        def test_columns(self, small_table):
            assert list(small_table.items()) == [('ID', [1, 2]), ('Name', ['Alpha', 'Beta'])]

    class TestAdd:
        def test_tables(self, small_table):
            other = Table(data=[[3, 'Gamma']], headers=['ID', 'Name'])
            combined = small_table + other
            assert combined.get_col('ID') == [1, 2, 3]
            assert len(small_table) == 2
            assert len(other) == 1

        def test_different_headers(self, small_table):
            with pytest.raises(ValueError):
                small_table + Table(data=[[True, 10]], headers=['Active', 'Value'])

        def test_other_type(self, small_table):
            with pytest.raises(TypeError):
                small_table + [3, 'Gamma']

        def test_formats(self, fmt_table):
            assert (fmt_table + fmt_table).formats == fmt_table.formats

        def test_copies_rows(self, small_table):
            other = Table(data=[[3, 'Gamma']], headers=['ID', 'Name'])
            combined = small_table + other
            combined['ID', 2] = 99
            assert other.get_col('ID') == [3]

    class TestIAdd:
        def test_table(self, small_table):
            small_table += Table(data=[[3, 'Gamma']], headers=['ID', 'Name'])
            assert small_table.get_col('ID') == [1, 2, 3]

        def test_self(self, small_table):
            small_table += small_table
            assert small_table.get_col('ID') == [1, 2, 1, 2]

        def test_row(self, small_table):
            small_table += [3, 'Gamma']
            small_table += {'ID': 4, 'Name': 'Delta'}
            assert small_table.get_col('ID') == [1, 2, 3, 4]

        def test_different_headers(self, small_table):
            with pytest.raises(ValueError):
                small_table += Table(data=[[True, 10]], headers=['Active', 'Value'])

        @pytest.mark.xfail(strict=True, reason="The other table's rows are appended, not copied")
        def test_copies_rows(self, small_table):
            other = Table(data=[[3, 'Gamma']], headers=['ID', 'Name'])
            small_table += other
            small_table['ID', 2] = 99
            assert other.get_col('ID') == [3]

    class TestOr:
        def test_tables(self, small_table):
            combined = small_table | Table(data=[[True], [False]], headers=['Active'])
            assert combined.headers == ['ID', 'Name', 'Active']
            assert combined.get_row(1) == [2, 'Beta', False]
            assert small_table.headers == ['ID', 'Name']

        def test_different_row_counts(self, small_table):
            with pytest.raises(ValueError):
                small_table | Table(data=[[True]], headers=['Active'])

        def test_other_type(self, small_table):
            with pytest.raises(TypeError):
                small_table | [True, False]

        def test_formats(self, fmt_table):
            other = Table(data=[[1], [2], [3]], headers=['E'], formats={'E': '.1f'})
            assert (fmt_table | other).formats == {'Value': '.2f', 'Count': '>8,d', 'E': '.1f'}

        def test_formats_without_headers(self):
            t1 = Table(data=[[1.0]], formats={0: '.1f'})
            t2 = Table(data=[[2.0]], formats={0: '.2f'})
            assert (t1 | t2).formats == {0: '.1f', 1: '.2f'}

    class TestIOr:
        def test_table(self, small_table):
            small_table |= Table(data=[[True], [False]], headers=['Active'])
            assert small_table.headers == ['ID', 'Name', 'Active']
            assert small_table.get_row(0) == [1, 'Alpha', True]

        def test_self(self, small_table):
            small_table |= small_table
            assert small_table.get_row(0) == [1, 'Alpha', 1, 'Alpha']

        def test_column(self, small_table):
            small_table |= {'Active': [True, False]}
            small_table |= 'Notes'
            assert small_table.headers == ['ID', 'Name', 'Active', 'Notes']
            assert len(small_table.get_col('Notes')) == 2

        def test_different_row_counts(self, small_table):
            with pytest.raises(ValueError):
                small_table |= Table(data=[[True]], headers=['Active'])

        def test_formats(self, fmt_table):
            fmt_table |= Table(data=[[1], [2], [3]], headers=['E'], formats={'E': '.1f'})
            assert fmt_table.formats == {'Value': '.2f', 'Count': '>8,d', 'E': '.1f'}

        def test_formats_without_headers(self):
            t1 = Table(data=[[1.0]], formats={0: '.1f'})
            t1 |= Table(data=[[2.0]], formats={0: '.2f'})
            assert t1.formats == {0: '.1f', 1: '.2f'}

    class TestDelRow:
        def test_index(self, sample_table):
            sample_table.del_row(2)
            sample_table.del_row(0)
            sample_table.del_row(-1)
            assert sample_table.get_col('Name') == ['Beta', 'Delta']

        @pytest.mark.parametrize('idx', [10, -10], ids=['past-end', 'before-start'])
        def test_out_of_range(self, sample_table, idx):
            with pytest.raises(IndexError):
                sample_table.del_row(idx)

    class TestDelCol:
        def test_name(self, sample_table):
            sample_table.del_col('Value')
            assert sample_table.headers == ['ID', 'Name', 'Active', 'Color']
            assert sample_table.get_row(2) == [3, 'Gamma', True, 'MISSING']

        def test_index(self, sample_table):
            sample_table.del_col(1)
            sample_table.del_col(-1)
            assert sample_table.headers == ['ID', 'Value', 'Active']

        @pytest.mark.parametrize(
            'idx, error',
            [(10, IndexError), (-10, IndexError), ('Other', KeyError)],
            ids=['past-end', 'before-start', 'name'],
        )
        def test_invalid(self, sample_table, idx, error):
            with pytest.raises(error):
                sample_table.del_col(idx)

        def test_format(self, fmt_table):
            fmt_table.del_col('Value')
            assert fmt_table.formats == {'Count': '>8,d'}

        def test_format_without_headers(self):
            table = Table(data=[[1.0, 2.0]], formats={1: '.1e'})
            table.del_col(0)
            assert table.formats == {0: '.1e'}

    class TestSub:
        def test_name(self, sample_table):
            result = sample_table - 'Value'
            assert result.headers == ['ID', 'Name', 'Active', 'Color']
            assert result.no_rows == 5
            assert sample_table.no_cols == 5

        @pytest.mark.parametrize('cols', [['ID', 'Active'], ('ID', 'Active')], ids=['list', 'tuple'])
        def test_names(self, sample_table, cols):
            result = sample_table - cols
            assert result.headers == ['Name', 'Value', 'Color']
            assert result.get_row(2) == ['Gamma', 30.7, 'MISSING']

        def test_unknown_name(self, sample_table):
            with pytest.raises(KeyError):
                sample_table - 'Other'

        def test_table(self, sample_table):
            result = sample_table - Table(headers=['ID', 'Label'])
            assert result.headers == ['Name', 'Value', 'Active', 'Color']

        def test_table_no_common_columns(self, sample_table):
            assert (sample_table - Table(headers=['A'])).headers == sample_table.headers

        def test_table_all_columns(self, sample_table):
            result = sample_table - Table(headers=sample_table.headers)
            assert result.shape == (0, 0)

        def test_other_type(self, sample_table):
            with pytest.raises(TypeError):
                sample_table - 1

        def test_formats(self, fmt_table):
            assert (fmt_table - 'Value').formats == {'Count': '>8,d'}

    class TestISub:
        def test_name(self, sample_table):
            original = sample_table
            sample_table -= 'Value'
            assert sample_table is original
            assert sample_table.headers == ['ID', 'Name', 'Active', 'Color']
            assert sample_table.get_row(2) == [3, 'Gamma', True, 'MISSING']

        def test_names(self, sample_table):
            sample_table -= ['ID', 'Active']
            assert sample_table.headers == ['Name', 'Value', 'Color']
            assert sample_table.get_row(0) == ['Alpha', 10.5, 'Red']

        def test_table(self, sample_table):
            sample_table -= Table(headers=['ID', 'Label'])
            assert sample_table.headers == ['Name', 'Value', 'Active', 'Color']
            sample_table -= Table(headers=['A'])
            assert sample_table.headers == ['Name', 'Value', 'Active', 'Color']

        def test_table_all_columns(self, sample_table):
            # The rows stay, without cells
            sample_table -= Table(headers=sample_table.headers)
            assert sample_table.headers == []
            assert sample_table.no_rows == 5

        def test_other_type(self, sample_table):
            with pytest.raises(TypeError):
                sample_table -= 1

        @pytest.mark.xfail(strict=True, reason='An unknown name raises ValueError from list.index()')
        def test_unknown_name(self, sample_table):
            with pytest.raises(KeyError):
                sample_table -= 'Other'


class TestOptionalDependencies:
    @pytest.mark.parametrize(
        'lib, call',
        [
            ('numpy', lambda t: t.to_numpy()),
            ('pandas', lambda t: t.to_pandas()),
            ('rich', lambda t: t.to_rich()),
            ('numpy', lambda t: Table.from_numpy([[1]])),
            ('pandas', lambda t: Table.from_pandas([[1]])),
        ],
        ids=['to-numpy', 'to-pandas', 'to-rich', 'from-numpy', 'from-pandas'],
    )
    def test_missing_names_the_extra(self, monkeypatch, lib, call):
        # A `None` entry in `sys.modules` makes the import fail
        monkeypatch.setitem(sys.modules, lib, None)
        monkeypatch.setitem(sys.modules, f'{lib}.table', None)
        with pytest.raises(ImportError, match=r'pip install xtl-core\['):
            call(Table(data=[[1, 2]], headers=['a', 'b']))


def test_package_exports():
    import xtl_core.containers
    from xtl_core.containers import tables

    for name in xtl_core.containers.__all__:
        assert getattr(xtl_core.containers, name) is getattr(tables, name)
