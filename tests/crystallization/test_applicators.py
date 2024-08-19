import pytest

import numpy as np

from xtl.crystallization.applicators import ConstantApplicator, GradientApplicator


class TestConstantApplicator:

    @pytest.mark.parametrize('value', [1.0, 2, 3.0])
    def test_init(self, value):
        ca = ConstantApplicator(value)
        value = float(value)
        assert ca.name.value == 'constant'
        assert ca.value == value
        assert ca.min_value == value
        assert ca.max_value == value

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize('value', ['1.0', [], {}])
    def test_init_fail_type_error(self, value):
        with pytest.raises(TypeError):
            ca = ConstantApplicator(value)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize('value', [-1.0, 0])
    def test_init_fail_value_error(self, value):
        with pytest.raises(ValueError):
            ca = ConstantApplicator(value)

    @pytest.mark.parametrize('shape', [(8, 12), (5, 1), (1, 7)])
    def test_apply(self, shape):
        ca = ConstantApplicator(1.0)
        data = ca.apply(shape)
        assert np.all(data == np.full(shape, 1.0).ravel())


class TestGradientApplicator:

    @pytest.mark.parametrize('min_value, max_value', [(1.0, 2.0), (2, 3), (3.0, 4)])
    def test_init(self, min_value, max_value):
        ga = GradientApplicator(min_value, max_value)
        assert ga.name.value == 'gradient'
        assert ga.min_value == float(min_value)
        assert ga.max_value == float(max_value)
        assert ga.application.value == 'horizontal'
        assert ga.scale.value == 'linear'
        assert not ga.reverse

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize('min_value, max_value', [(1.0, '2.0'), ('2', 3), (3.0, [])])
    def test_init_fail_type_error(self, min_value, max_value):
        with pytest.raises(TypeError):
            ga = GradientApplicator(min_value, max_value)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize('min_value, max_value', [(-1.0, 2.0), (2, 0), (3.0, -4)])
    def test_init_fail_value_error(self, min_value, max_value):
        with pytest.raises(ValueError):
            ga = GradientApplicator(min_value, max_value)

    @pytest.mark.parametrize('application, scale, reverse', [
        ('horizontal', 'linear', False),
        ('vertical', 'logarithmic', True),
        ('continuous', 'linear', False),
    ])
    def test_init_params(self, application, scale, reverse):
        ga = GradientApplicator(1.0, 2.0, application, scale, reverse)
        assert ga.application.value == application
        assert ga.scale.value == scale
        assert ga.reverse == reverse

    @pytest.mark.parametrize('shape, application, scale, reverse, expected', [
        ((8, 12), 'continuous', 'linear', True, np.linspace(2.0, 1.0, 96)),
        ((1, 5), 'horizontal', 'logarithmic', False, np.geomspace(1.0, 2.0, 5)),
        ((3, 6), 'vertical', 'linear', True, np.tile(np.linspace(2.0, 1.0, 3), (6, 1)).T.ravel()),
    ])
    def test_apply(self, shape, application, scale, reverse, expected):
        ga = GradientApplicator(min_value=1.0, max_value=2.0, application=application, scale=scale, reverse=reverse)
        data = ga.apply(shape=shape)
        assert np.all(data == expected)

