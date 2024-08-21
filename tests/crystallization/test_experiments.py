import pytest

import numpy as np

from xtl.crystallization.experiments import CrystallizationExperiment

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
        assert ce._reagents_map is None
        assert ce._shape == expected
        assert ce._ndim == len(shape) if isinstance(shape, tuple) else 1
        assert ce._pH.shape == expected

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
        assert np.all(indices == np.sort(expected))

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
        assert np.all(indices == np.sort(expected))

