import pytest

import numpy as np

from xtl.crystallization.experiments import CrystallizationExperiment
from xtl.io.npx import npx_load

class TestCrystallizationExperiment:

    @pytest.mark.parametrize(
        'shape,   expected', [
        (8,       (8, )),
        ((8, 12), (8, 12)),
        ((5, 1),  (5, 1)),
        ((1, 7),  (1, 7))
    ])
    def test_init(self, shape, expected):
        ce = CrystallizationExperiment(shape=shape)
        assert ce._data is None
        assert ce._volumes is None
        assert ce._pH is not None
        assert ce._reagents == []
        assert ce._shape == expected
        assert ce._ndim == len(shape) if isinstance(shape, tuple) else 1
        assert ce._pH.shape == expected

    @pytest.mark.parametrize(
        'index,             expected', [
        (0,                 (0, 0)),
        (1,                 (0, 1)),
        (84,                (7, 0)),
        (77,                (6, 5)),
        (95,                (7, 11)),
        (np.array([0, 84]), np.array([[0, 0], [7, 0]])),
        (np.array([1, 77]), np.array([[0, 1], [6, 5]])),
        ])
    def test_index_1D_to_2D(self, index, expected):
        ce = CrystallizationExperiment(shape=(8, 12))
        if isinstance(index, int):
            assert ce._index_1D_to_2D(index) == expected
        elif isinstance(index, np.ndarray):
            assert np.array_equal(ce._index_1D_to_2D(index), expected, equal_nan=True)

    @pytest.mark.parametrize(
        'index,                      expected', [
        ((0, 0),                     0),
        ((0, 1),                     1),
        ((7, 0),                     84),
        ((6, 5),                     77),
        ((7, 11),                    95),
        (np.array([[0, 0], [7, 0]]), np.array([0, 84])),
        (np.array([[0, 1], [6, 5]]), np.array([1, 77])),
        ])
    def test_index_2D_to_1D(self, index, expected):
        ce = CrystallizationExperiment(shape=(8, 12))
        if isinstance(index, tuple):
            assert ce._index_2D_to_1D(row=index[0], col=index[1]) == expected
        elif isinstance(index, np.ndarray):
            assert np.array_equal(ce._index_2D_to_1D(row=index[:, 0], col=index[:, 1]), expected, equal_nan=True)

    @pytest.mark.parametrize(
        'shape,   location,     expected', [
        ((8, 12), 'everywhere', np.arange(0, 96)),
        ((8, 12), 'all',        np.arange(0, 96)),
        ((8, 12), 'row1',       np.arange(0, 12)),  # single line
        ((8, 12), 'col2',       np.array([ 1, 13, 25, 37, 49, 61, 73, 85])),
        ((8, 12), 'row9',       np.array([])),  # out of bounds
        ((8, 12), 'col13',      np.array([])),
        ((8, 12), 'row4-8',     np.arange(36, 96)),  # line range
        ((8, 12), 'col3-7',     np.array([ 2,  3,  4,  5,  6,   14, 15, 16, 17, 18,   26, 27, 28, 29, 30,
                                          38, 39, 40, 41, 42,   50, 51, 52, 53, 54,   62, 63, 64, 65, 66,
                                          74, 75, 76, 77, 78,   86, 87, 88, 89, 90])),
        ((8, 12), 'row1,3',     np.hstack((np.arange(0, 12), np.arange(24, 36)))),  # comma separated lines
        ((8, 12), 'col2,4',     np.array([ 1, 13, 25, 37, 49, 61, 73, 85,
                                           3, 15, 27, 39, 51, 63, 75, 87])),
        ((8, 12), 'row1-3,5-7', np.hstack((np.arange(0, 36), np.arange(48, 84)))),  # comma separated ranges
        ((8, 12), 'col1-11,12', np.arange(0, 96)),
        ((8, 12), 'cell1',      np.array([0])),
        ((8, 12), 'cell2-7',    np.arange(1, 7)),
        ((8, 12), 'cell8,76',   np.array([7, 75])),
        ((8, 12), 'cell97',     np.array([])),
        ((8, 12), 'cell1-9,15', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 14])),
    ])
    def test_location_to_indices_str(self, shape, location, expected):
        ce = CrystallizationExperiment(shape=shape)
        indices = ce._location_to_indices(location)
        assert np.array_equal(indices, np.sort(expected), equal_nan=True)

    @pytest.mark.parametrize(
        'shape,   location,             expected', [
        ((8, 12), ['cell1', 'cell3'],   np.array([0, 2])),
        ((8, 12), [5, 12],              np.array([4, 11])),
        ((8, 12), [(1, 1), (8, 12)],    np.array([0, 95])),
        ((8, 12), [(1, 1), 'row2', 25], np.hstack([np.array([0, 24]), np.arange(12, 24)])),
    ])
    def test_location_to_indices_list(self, shape, location, expected):
        ce = CrystallizationExperiment(shape=shape)
        indices = ce._location_to_indices(location)
        assert np.array_equal(indices, np.sort(expected), equal_nan=True)

    @pytest.mark.parametrize(
        'shape,   location,    expected_map_i,        expected_mask', [
        ((8, 12), 'cell1',     np.array([0]),         np.array([1.0]).reshape((1, 1))),
        ((8, 12), 'row1',      np.arange(0, 12),      np.full(12, fill_value=1.0).reshape((1, 12))),
        ((8, 12), 'col2',      np.arange(8) * 12 + 1, np.full(8, fill_value=1.0).reshape((8, 1))),
        ((8, 12), 'cell1-96',  np.arange(96),         np.full(96, fill_value=1.0).reshape((8, 12))),
        ((8, 12), 'cell14,81', np.vstack([np.arange(8) + 13 + 12 * a for a in range(0, 6)]),
                                                      np.vstack([[1.0] + [np.nan] * 7,
                                                                np.tile(np.full(8, np.nan), (4, 1)),
                                                                [np.nan] * 7 + [1.0]]))
    ])
    def test_location_to_map(self, shape, location, expected_map_i, expected_mask):
        ce = CrystallizationExperiment(shape=shape)
        location_map, mask = ce._location_to_map(location)
        expected_map = np.full(shape, fill_value=False).ravel()
        expected_map[expected_map_i] = True
        assert location_map.shape == ce.shape
        assert np.array_equal(location_map, expected_map.reshape(shape), equal_nan=True)
        assert np.array_equal(mask, expected_mask, equal_nan=True)

    @pytest.mark.parametrize('npx_file', ['data/reshape_data_ex1.npx',
                                          'data/reshape_data_ex2.npx',
                                          'data/reshape_data_ex3.npx'])
    def test_reshape_data(self, npx_file):
        npx = npx_load(npx_file)
        shape = tuple(npx.data['shape'])
        (r_min, c_min), (r_max, c_max) = npx.data['indices']
        data = npx.data['data']
        mask = npx.data['mask']
        location_map = npx.data['location_map']
        data_reshaped_expected = npx.data['data_reshaped']
        mask_reshaped_expected = npx.data['mask_reshaped']

        ce = CrystallizationExperiment(shape=shape)
        data_reshaped, mask_reshaped = ce._reshape_data(array=data, location_map=location_map, mask=mask)
        assert np.array_equal(data_reshaped, data_reshaped_expected, equal_nan=True)
        assert np.array_equal(mask_reshaped, mask_reshaped_expected, equal_nan=True)


    @pytest.mark.parametrize('npx_file', ['data/reshape_data_ex1.npx',
                                          'data/reshape_data_ex2.npx',
                                          'data/reshape_data_ex3.npx'])
    def test_reshape_data_without_mask(self, npx_file):
        npx = npx_load(npx_file)
        shape = tuple(npx.data['shape'])
        (r_min, c_min), (r_max, c_max) = npx.data['indices']
        data = npx.data['data']
        location_map = npx.data['location_map']
        # data_reshaped_expected = npx.data['data_reshaped']
        mask_reshaped_expected = location_map.astype(float)
        mask_reshaped_expected[np.where(mask_reshaped_expected == 0.)] = np.nan

        ce = CrystallizationExperiment(shape=shape)
        data_reshaped, mask_reshaped = ce._reshape_data(array=data, location_map=location_map, mask=None)
        # assert np.array_equal(data_reshaped, data_reshaped_expected, equal_nan=True)
        assert np.array_equal(mask_reshaped, mask_reshaped_expected, equal_nan=True)
