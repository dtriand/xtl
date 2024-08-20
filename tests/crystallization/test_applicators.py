import pytest

import numpy as np

from xtl.crystallization.applicators import ConstantApplicator, GradientApplicator, StepFixedApplicator


class TestConstantApplicator:

    @pytest.mark.parametrize('value', [1.0, 2, 3.0])
    def test_init(self, value):
        ca = ConstantApplicator(value=value)
        value = float(value)
        assert ca.name.value == 'constant'
        assert ca.value == value
        assert ca.min_value == value
        assert ca.max_value == value

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize('value', ['1.0', [], {}])
    def test_init_fail_type_error(self, value):
        with pytest.raises(TypeError):
            ca = ConstantApplicator(value=value)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize('value', [-1.0, 0])
    def test_init_fail_value_error(self, value):
        with pytest.raises(ValueError):
            ca = ConstantApplicator(value=value)

    @pytest.mark.parametrize('shape', [(8, 12), (5, 1), (1, 7)])
    def test_apply(self, shape):
        ca = ConstantApplicator(value=1.0)
        data = ca.apply(shape=shape)
        assert np.all(data == np.full(shape, 1.0).ravel())


class TestGradientApplicator:

    @pytest.mark.parametrize('min_value, max_value', [(1.0, 2.0), (2, 3), (3.0, 4)])
    def test_init(self, min_value, max_value):
        ga = GradientApplicator(min_value=min_value, max_value=max_value)
        assert ga.name.value == 'gradient'
        assert ga.min_value == float(min_value)
        assert ga.max_value == float(max_value)
        assert ga.method.value == 'horizontal'
        assert ga.scale.value == 'linear'
        assert not ga.reverse

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize('min_value, max_value', [(1.0, '2.0'), ('2', 3), (3.0, [])])
    def test_init_fail_type_error(self, min_value, max_value):
        with pytest.raises(TypeError):
            ga = GradientApplicator(min_value=min_value, max_value=max_value)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize('min_value, max_value', [(-1.0, 2.0), (2, 0), (3.0, -4)])
    def test_init_fail_value_error(self, min_value, max_value):
        with pytest.raises(ValueError):
            ga = GradientApplicator(min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize('method, scale, reverse', [
        ('horizontal', 'linear', False),
        ('vertical', 'logarithmic', True),
        ('continuous', 'linear', False),
    ])
    def test_init_params(self, method, scale, reverse):
        ga = GradientApplicator(min_value=1.0, max_value=2.0, method=method, scale=scale, reverse=reverse)
        assert ga.method.value == method
        assert ga.scale.value == scale
        assert ga.reverse == reverse

    @pytest.mark.parametrize('shape, method, scale, reverse, expected', [
        ((8, 12), 'continuous', 'linear', True, np.linspace(2.0, 1.0, 96)),
        ((1, 5), 'horizontal', 'logarithmic', False, np.geomspace(1.0, 2.0, 5)),
        ((3, 6), 'vertical', 'linear', True, np.tile(np.linspace(2.0, 1.0, 3), (6, 1)).T.ravel()),
        ((2, 3), 'continuous', 'linear', False, np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])),
        ((3, 2), 'vertical', 'linear', True, np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0]))
    ])
    def test_apply(self, shape, method, scale, reverse, expected):
        ga = GradientApplicator(min_value=1.0, max_value=2.0, method=method, scale=scale, reverse=reverse)
        data = ga.apply(shape=shape)
        assert np.all(data == expected)


class TestStepFixedApplicator:

    @pytest.mark.parametrize('start_value, step', [(1.0, 2.0), (0, 3), (3.0, 4)])
    def test_init(self, start_value, step):
        sfa = StepFixedApplicator(start_value=start_value, step=step)
        assert sfa.name.value == 'step_fixed'
        assert sfa.min_value == float(start_value)
        assert sfa.max_value is None
        assert sfa.step == float(step)
        assert sfa.method.value == 'horizontal'
        assert not sfa.reverse

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize('start_value, step', [(1.0, '2.0'), ('2', 3), (3.0, [])])
    def test_init_fail_type_error(self, start_value, step):
        with pytest.raises(TypeError):
            sfa = StepFixedApplicator(start_value=start_value, step=step)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize('start_value, step', [(-1.0, 2.0), (2, 0)])
    def test_init_fail_value_error(self, start_value, step):
        with pytest.raises(ValueError):
            sfa = StepFixedApplicator(start_value=start_value, step=step)

    def test_init_negative_step(self):
        sfa = StepFixedApplicator(start_value=1.0, step=-2.0)
        assert sfa.max_value == 1.0
        assert sfa.step == 2.0
        assert sfa.reverse

    @pytest.mark.parametrize('shape, method, reverse, expected', [
        ((8, 12), 'continuous', False, np.arange(start=10.0, stop=10.0 + 0.5 * 96, step=0.5)),
        ((1, 5), 'horizontal', False, np.array([10.0, 10.5, 11.0, 11.5, 12.0])),
        ((3, 6), 'vertical', True, np.tile([10., 9.5, 9.0], (6, 1)).T.ravel()),
        ((2, 3), 'continuous', True, np.array([10., 9.5, 9.0, 8.5, 8.0, 7.5]))
    ])
    def test_apply(self, shape, method, reverse, expected):
        sfa = StepFixedApplicator(start_value=10.0, step=0.5, method=method, reverse=reverse)
        data = sfa.apply(shape=shape)
        assert np.all(data == expected)

    @pytest.mark.xfail(raises=ValueError)
    def test_apply_negative_min(self):
        with pytest.raises(ValueError):
            sfa = StepFixedApplicator(start_value=1.0, step=-0.5)
            sfa.apply(shape=(8, 12))
