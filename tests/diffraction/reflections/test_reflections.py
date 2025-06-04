import copy
import pytest
import numpy as np
import pandas as pd
import gemmi
import reciprocalspaceship as rs

from xtl.diffraction.reflections.reflections import (
    _ReflectionsBase,
    ReflectionsData,
    ReflectionsCollection,
    MTZ_DTYPES
)
from xtl.diffraction.reflections.metadata import ReflectionsMetadata


class TestReflectionsBase:
    """Tests for the _ReflectionsBase class"""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing"""
        # Create a simple dataset with HKL indices and one data column
        df = rs.DataSet({
            'H': [1, 2, 3, 4, 5],
            'K': [0, 0, 0, 0, 0],
            'L': [0, 0, 0, 0, 0],
            'F': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        # Set space group and unit cell
        df.spacegroup = gemmi.SpaceGroup('P 1')
        df.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
        df.set_index(['H', 'K', 'L'], inplace=True)
        return df

    @pytest.fixture
    def base_reflections(self, mock_dataset):
        """Create a _ReflectionsBase instance for testing"""
        return _ReflectionsBase(data=mock_dataset)

    def test_init(self, mock_dataset):
        """Test initialization of _ReflectionsBase"""
        reflections = _ReflectionsBase(data=mock_dataset)
        assert reflections._rs is not None
        assert reflections._rs.shape == mock_dataset.shape
        assert reflections._metadata is None

    def test_get_dtype(self, base_reflections):
        """Test _get_dtype method"""
        # Test with string type
        dtype = base_reflections._get_dtype('F')
        assert dtype.__class__.__name__ == 'StructureFactorAmplitudeDtype'

        # Test with MTZDtype directly
        f_dtype = MTZ_DTYPES['F']
        dtype = base_reflections._get_dtype(f_dtype)
        assert dtype == f_dtype

        # Test with iterable
        dtypes = base_reflections._get_dtype(['F', 'J'])
        assert len(dtypes) == 2
        assert dtypes[0].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtypes[1].__class__.__name__ == 'IntensityDtype'

        # Test with dict
        dtype_dict = base_reflections._get_dtype({'col1': 'F', 'col2': 'J'})
        assert len(dtype_dict) == 2
        assert dtype_dict['col1'].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtype_dict['col2'].__class__.__name__ == 'IntensityDtype'

        # Test with pandas Series
        series = pd.Series(['F', 'J', 'Q'])
        dtypes = base_reflections._get_dtype(series)
        assert isinstance(dtypes, list)
        assert len(dtypes) == 3
        assert dtypes[0].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtypes[1].__class__.__name__ == 'IntensityDtype'
        assert dtypes[2].__class__.__name__ == 'StandardDeviationDtype'

        # Test with pandas Index
        index = pd.Index(['F', 'J', 'Q'])
        dtypes = base_reflections._get_dtype(index)
        assert isinstance(dtypes, list)
        assert len(dtypes) == 3
        assert dtypes[0].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtypes[1].__class__.__name__ == 'IntensityDtype'
        assert dtypes[2].__class__.__name__ == 'StandardDeviationDtype'

        # Test with pandas Series containing mixed types
        series = pd.Series(['F', MTZ_DTYPES['J'], 'Q'])
        dtypes = base_reflections._get_dtype(series)
        assert isinstance(dtypes, list)
        assert len(dtypes) == 3
        assert dtypes[0].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtypes[1].__class__.__name__ == 'IntensityDtype'
        assert dtypes[2].__class__.__name__ == 'StandardDeviationDtype'

        # Test with pandas Series with invalid type
        series_with_invalid = pd.Series(['F', 'INVALID_TYPE', 'Q'])
        with pytest.raises(ValueError):
            base_reflections._get_dtype(series_with_invalid)

        # Test with invalid type
        with pytest.raises(ValueError):
            base_reflections._get_dtype('INVALID_TYPE')

        # Test with invalid input type
        with pytest.raises(TypeError):
            base_reflections._get_dtype(123)


    def test_dropna(self, base_reflections, mock_dataset):
        """Test dropna method"""
        # Create a copy with some NaN values
        df_with_nan = mock_dataset.copy()
        df_with_nan.loc[(3, 0, 0), 'F'] = np.nan

        reflections = _ReflectionsBase(data=df_with_nan)
        assert reflections.no_reflections == 5

        # Test without inplace
        new_reflections = reflections.dropna(inplace=False)
        assert reflections.no_reflections == 5  # Original unchanged
        assert new_reflections.no_reflections == 4  # New has one less

        # Test with inplace
        reflections.dropna(inplace=True)
        assert reflections.no_reflections == 4

    def test_shape_property(self, base_reflections):
        """Test shape property"""
        assert base_reflections.shape == (5, 1)

    def test_ndim_property(self, base_reflections):
        """Test ndim property"""
        assert base_reflections.ndim == 2

    def test_no_reflections_property(self, base_reflections):
        """Test no_reflections property"""
        assert base_reflections.no_reflections == 5

    def test_no_columns_property(self, base_reflections):
        """Test no_columns property"""
        assert base_reflections.no_columns == 1

    def test_hkls_property(self, base_reflections):
        """Test hkls property"""
        hkls = base_reflections.hkls
        assert isinstance(hkls, np.ndarray)
        assert hkls.shape == (5, 3)
        assert np.array_equal(hkls[:, 0], np.array([1, 2, 3, 4, 5]))
        assert np.all(hkls[:, 1] == 0)
        assert np.all(hkls[:, 2] == 0)

    def test_space_group_property(self, base_reflections):
        """Test space_group property and setter"""
        assert base_reflections.space_group.hm == 'P 1'

        # Test setter
        base_reflections.space_group = 'P 21 21 21'
        assert base_reflections.space_group.hm == 'P 21 21 21'

        # Test setter with SpaceGroup object
        base_reflections.space_group = gemmi.SpaceGroup('C 1 2 1')
        assert base_reflections.space_group.hm == 'C 1 2 1'

        # Test setter with number
        base_reflections.space_group = 1
        assert base_reflections.space_group.hm == 'P 1'

    def test_unit_cell_property(self, base_reflections):
        """Test unit_cell property and setter"""
        uc = base_reflections.unit_cell
        assert uc.a == 10.0
        assert uc.b == 10.0
        assert uc.c == 10.0
        assert uc.alpha == 90.0
        assert uc.beta == 90.0
        assert uc.gamma == 90.0

        # Test setter with UnitCell object
        new_uc = gemmi.UnitCell(20, 20, 20, 90, 90, 90)
        base_reflections.unit_cell = new_uc
        assert base_reflections.unit_cell.a == 20.0

        # Test setter with list
        base_reflections.unit_cell = [30, 30, 30, 90, 90, 90]
        assert base_reflections.unit_cell.a == 30.0

        # Test setter with tuple
        base_reflections.unit_cell = (40, 40, 40, 90, 90, 90)
        assert base_reflections.unit_cell.a == 40.0

        # Test setter with numpy array
        base_reflections.unit_cell = np.array([50, 50, 50, 90, 90, 90])
        assert base_reflections.unit_cell.a == 50.0

    def test_is_merged_property(self, base_reflections):
        """Test is_merged property and setter"""
        # Default should be None
        assert base_reflections.is_merged is None

        # Test setter
        base_reflections.is_merged = True
        assert base_reflections.is_merged is True

        base_reflections.is_merged = False
        assert base_reflections.is_merged is False

        # Test setter with invalid value
        with pytest.raises(TypeError):
            base_reflections.is_merged = "not a boolean"

    def test_resolution_properties(self, base_reflections):
        """Test resolution-related properties"""
        # Calculate expected d-spacings for the test reflections
        # For [1,0,0], [2,0,0], etc. with a=10 Å, d = a/h (for cubic cell)
        expected_d_values = np.array([10.0, 5.0, 3.333, 2.5, 2.0])

        # Test min_d property
        assert pytest.approx(base_reflections.min_d) == 2.0

        # Test max_d property
        assert pytest.approx(base_reflections.max_d) == 10.0

        # Test range_d property
        d_range = base_reflections.range_d
        assert pytest.approx(d_range[0]) == 10.0
        assert pytest.approx(d_range[1]) == 2.0

        # Test resolution_high property (alias for min_d)
        assert pytest.approx(base_reflections.resolution_high) == 2.0

        # Test resolution_low property (alias for max_d)
        assert pytest.approx(base_reflections.resolution_low) == 10.0

        # Test resolution_range property (alias for range_d)
        res_range = base_reflections.resolution_range
        assert pytest.approx(res_range[0]) == 10.0
        assert pytest.approx(res_range[1]) == 2.0

    def test_get_array_d(self, base_reflections):
        """Test get_array_d method"""
        d_values = base_reflections.get_array_d()
        assert isinstance(d_values, np.ndarray)
        assert d_values.shape == (5,)
        expected_d_values = np.array([10.0, 5.0, 3.333, 2.5, 2.0])
        np.testing.assert_allclose(d_values, expected_d_values, rtol=1e-3)

    def test_1_d_properties(self, base_reflections):
        """Test 1/d-related properties"""
        # Calculate expected 1/d values for the test reflections
        expected_1_d_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Test min_1_d property
        assert pytest.approx(base_reflections.min_1_d) == 0.1

        # Test max_1_d property
        assert pytest.approx(base_reflections.max_1_d) == 0.5

        # Test range_1_d property
        d_range = base_reflections.range_1_d
        assert pytest.approx(d_range[0]) == 0.1
        assert pytest.approx(d_range[1]) == 0.5

    def test_get_array_1_d(self, base_reflections):
        """Test get_array_1_d method"""
        d_values = base_reflections.get_array_1_d()
        assert isinstance(d_values, np.ndarray)
        assert d_values.shape == (5,)
        expected_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(d_values, expected_values, rtol=1e-3)

    def test_1_d2_properties(self, base_reflections):
        """Test 1/d^2-related properties"""
        # Calculate expected 1/d^2 values for the test reflections
        expected_1_d2_values = np.array([0.01, 0.04, 0.09, 0.16, 0.25])

        # Test min_1_d2 property
        assert pytest.approx(base_reflections.min_1_d2) == 0.01

        # Test max_1_d2 property
        assert pytest.approx(base_reflections.max_1_d2) == 0.25

        # Test range_1_d2 property
        d_range = base_reflections.range_1_d2
        assert pytest.approx(d_range[0]) == 0.01
        assert pytest.approx(d_range[1]) == 0.25

    def test_get_array_1_d2(self, base_reflections):
        """Test get_array_1_d2 method"""
        d_values = base_reflections.get_array_1_d2()
        assert isinstance(d_values, np.ndarray)
        assert d_values.shape == (5,)
        expected_values = np.array([0.01, 0.04, 0.09, 0.16, 0.25])
        np.testing.assert_allclose(d_values, expected_values, rtol=1e-3)

    def test_min_max_hkl(self, base_reflections):
        """Test min_hkl and max_hkl properties"""
        # min_hkl should be the reflection with highest d-spacing (lowest resolution)
        min_hkl = base_reflections.min_hkl
        assert np.array_equal(min_hkl, [1, 0, 0])

        # max_hkl should be the reflection with lowest d-spacing (highest resolution)
        max_hkl = base_reflections.max_hkl
        assert np.array_equal(max_hkl, [5, 0, 0])


class TestReflectionsData:
    """Tests for the ReflectionsData class"""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing"""
        # Create a simple dataset with HKL indices and one data column
        df = rs.DataSet({
            'H': [1, 2, 3, 4, 5],
            'K': [0, 0, 0, 0, 0],
            'L': [0, 0, 0, 0, 0],
            'F': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        # Set space group and unit cell
        df.spacegroup = gemmi.SpaceGroup('P 1')
        df.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
        df.set_index(['H', 'K', 'L'], inplace=True)
        # Set dtype to structure factor amplitude
        df = df.astype({'F': MTZ_DTYPES['F']})
        return df

    @pytest.fixture
    def reflections_data(self, mock_dataset):
        """Create a ReflectionsData instance for testing"""
        return ReflectionsData(data=mock_dataset)

    def test_init(self, mock_dataset):
        """Test initialization of ReflectionsData"""
        reflections = ReflectionsData(data=mock_dataset)
        assert reflections._rs is not None
        assert reflections._rs.shape == mock_dataset.shape
        assert reflections._metadata is None

        # Test initialization with invalid shape (more than one column)
        df_multi = mock_dataset.copy()
        df_multi['SIGF'] = 1.0
        with pytest.raises(ValueError):
            ReflectionsData(data=df_multi)

    def test_dtype_property(self, reflections_data):
        """Test dtype property"""
        dtype = reflections_data.dtype
        assert dtype.__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtype.mtztype == 'F'

    def test_as_type(self, reflections_data):
        """Test as_type method"""
        # Test with MTZDtype directly
        reflections_data.as_type(MTZ_DTYPES['J'])
        assert reflections_data.dtype.__class__.__name__ == 'IntensityDtype'
        assert reflections_data.dtype.mtztype == 'J'

        # Test with MTZ type
        reflections_data.as_type('F')
        assert reflections_data.dtype.__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert reflections_data.dtype.mtztype == 'F'

        # Test with copy
        reflections_copy = copy.deepcopy(reflections_data)
        reflections_copy.as_type('Q', copy=True)
        assert reflections_copy.dtype.__class__.__name__ == 'StandardDeviationDtype'
        assert reflections_copy.dtype.mtztype == 'Q'

        # Test with invalid type
        with pytest.raises(ValueError):
            reflections_data.as_type('INVALID_TYPE')

    def test_mtz_type_property(self, reflections_data):
        """Test mtz_type property"""
        assert reflections_data.mtz_type == 'F'

    def test_column_property(self, reflections_data):
        """Test column property"""
        assert reflections_data.column[0] == 'F'

    def test_label_property(self, reflections_data):
        """Test label property"""
        assert reflections_data.label == 'F'

    def test_values_property(self, reflections_data):
        """Test values property"""
        values = reflections_data.values
        assert isinstance(values, rs.DataSeries)
        assert values.shape == (5,)
        assert values.iloc[0] == 10.0
        assert values.iloc[-1] == 50.0

    def test_from_rs(self, mock_dataset):
        """Test from_rs classmethod"""
        # Test with valid input
        reflections = ReflectionsData.from_rs(mock_dataset, 'F')
        assert reflections.label == 'F'
        assert reflections.no_reflections == 5

        # Test with non-DataSet input
        with pytest.raises(TypeError):
            ReflectionsData.from_rs({}, 'F')

        # Test with invalid label
        with pytest.raises(ValueError):
            ReflectionsData.from_rs(mock_dataset, 'INVALID_LABEL')


class TestReflectionsCollection:
    """Tests for the ReflectionsCollection class"""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing"""
        # Create a simple dataset with HKL indices and multiple data columns
        df = rs.DataSet({
            'H': [1, 2, 3, 4, 5],
            'K': [0, 0, 0, 0, 0],
            'L': [0, 0, 0, 0, 0],
            'F': [10.0, 20.0, 30.0, 40.0, 50.0],
            'SIGF': [1.0, 2.0, 3.0, 4.0, 5.0],
            'J': [100.0, 400.0, 900.0, 1600.0, 2500.0]
        })
        # Set space group and unit cell
        df.spacegroup = gemmi.SpaceGroup('P 1')
        df.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
        df.set_index(['H', 'K', 'L'], inplace=True)
        # Set dtypes
        df = df.astype({
            'F': MTZ_DTYPES['F'],
            'SIGF': MTZ_DTYPES['Q'],
            'J': MTZ_DTYPES['J']
        })
        return df

    @pytest.fixture
    def reflections_collection(self, mock_dataset):
        """Create a ReflectionsCollection instance for testing"""
        return ReflectionsCollection(data=mock_dataset)

    def test_init(self, mock_dataset):
        """Test initialization of ReflectionsCollection"""
        reflections = ReflectionsCollection(data=mock_dataset)
        assert reflections._rs is not None
        assert reflections._rs.shape == mock_dataset.shape
        assert reflections._metadata is None

    def test_metadata_property(self, reflections_collection):
        """Test metadata property"""
        assert reflections_collection.metadata is None

    def test_dtypes_property(self, reflections_collection):
        """Test dtypes property"""
        dtypes = reflections_collection.dtypes
        assert isinstance(dtypes, pd.Series)
        assert dtypes.shape == (3,)
        assert dtypes['F'].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert dtypes['SIGF'].__class__.__name__ == 'StandardDeviationDtype'
        assert dtypes['J'].__class__.__name__ == 'IntensityDtype'

    def test_as_type(self, reflections_collection):
        """Test as_type method"""
        # Test with dict of MTZDtype objects
        reflections_collection.as_type({
            'F': MTZ_DTYPES['J'],
            'SIGF': MTZ_DTYPES['Q'],
            'J': MTZ_DTYPES['F']
        })
        assert reflections_collection.dtypes['F'].__class__.__name__ == 'IntensityDtype'
        assert reflections_collection.dtypes['SIGF'].__class__.__name__ == 'StandardDeviationDtype'
        assert reflections_collection.dtypes['J'].__class__.__name__ == 'StructureFactorAmplitudeDtype'

        # Test with dict of string type codes
        reflections_collection.as_type({
            'F': 'F',
            'SIGF': 'Q',
            'J': 'J'
        })
        assert reflections_collection.dtypes['F'].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert reflections_collection.dtypes['SIGF'].__class__.__name__ == 'StandardDeviationDtype'
        assert reflections_collection.dtypes['J'].__class__.__name__ == 'IntensityDtype'

        # Test with iterable of MTZDtype objects
        reflections_copy = copy.deepcopy(reflections_collection)
        reflections_copy.as_type([
            MTZ_DTYPES['F'],
            MTZ_DTYPES['F'],
            MTZ_DTYPES['J']
        ], copy=True)
        assert reflections_copy.dtypes['F'].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert reflections_copy.dtypes['SIGF'].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert reflections_copy.dtypes['J'].__class__.__name__ == 'IntensityDtype'

        # Test with iterable of string type codes
        reflections_copy = copy.deepcopy(reflections_collection)
        reflections_copy.as_type(['J', 'F', 'Q'], copy=True)
        assert reflections_copy.dtypes['F'].__class__.__name__ == 'IntensityDtype'
        assert reflections_copy.dtypes['SIGF'].__class__.__name__ == 'StructureFactorAmplitudeDtype'
        assert reflections_copy.dtypes['J'].__class__.__name__ == 'StandardDeviationDtype'

        # Test with wrong number of dtypes
        with pytest.raises(ValueError):
            reflections_collection.as_type([MTZ_DTYPES['F']])

        # Test with invalid input type
        with pytest.raises(TypeError):
            reflections_collection.as_type(123)

        # Test with invalid type code
        with pytest.raises(ValueError):
            reflections_collection.as_type(['F', 'J', 'INVALID_TYPE'])

    def test_mtz_types_property(self, reflections_collection):
        """Test mtz_types property"""
        mtz_types = reflections_collection.mtz_types
        assert isinstance(mtz_types, tuple)
        assert len(mtz_types) == 3
        assert mtz_types[0] == 'F'
        assert mtz_types[1] == 'Q'
        assert mtz_types[2] == 'J'

    def test_columns_property(self, reflections_collection):
        """Test columns property"""
        columns = reflections_collection.columns
        assert isinstance(columns, pd.Index)
        assert columns.tolist() == ['F', 'SIGF', 'J']

    def test_labels_property(self, reflections_collection):
        """Test labels property"""
        labels = reflections_collection.labels
        assert isinstance(labels, tuple)
        assert labels == ('F', 'SIGF', 'J')

    def test_values_property(self, reflections_collection):
        """Test values property"""
        values = reflections_collection.values
        assert isinstance(values, rs.DataSet)
        assert values.shape == (5, 3)
        assert 'F' in values.columns
        assert 'SIGF' in values.columns
        assert 'J' in values.columns

    def test_from_rs(self, mock_dataset):
        """Test from_rs classmethod"""
        # Test with all columns
        reflections = ReflectionsCollection.from_rs(mock_dataset)
        assert reflections.no_columns == 3
        assert reflections.labels == ('F', 'SIGF', 'J')

        # Test with specific columns
        reflections = ReflectionsCollection.from_rs(mock_dataset, labels=['F', 'J'])
        assert reflections.no_columns == 2
        assert reflections.labels == ('F', 'J')

        # Test with non-DataSet input
        with pytest.raises(TypeError):
            ReflectionsCollection.from_rs({})

        # Test with invalid label
        with pytest.raises(ValueError):
            ReflectionsCollection.from_rs(mock_dataset, labels=['F', 'INVALID_LABEL'])

        # Test with non-iterable labels
        with pytest.raises(TypeError):
            ReflectionsCollection.from_rs(mock_dataset, labels=123)
