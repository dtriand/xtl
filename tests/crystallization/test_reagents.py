import pytest

from xtl.crystallization.reagents import Reagent, ReagentWV, ReagentVV, Buffer


class TestReagent:

    @pytest.mark.parametrize('concentration', [1.0, 2, 3.0])
    def test_init(self, concentration):
        r = Reagent(name='test', concentration=1.0)
        assert r.name == 'test'
        assert r.concentration == 1.0
        assert r.solubility is None
        assert r.unit == 'M'
        assert r.fmt_str is None

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize('concentration, solubility', [[1.0, '2.0'], ['2', 3]])
    def test_init_fail_type_error(self, concentration, solubility):
        with pytest.raises(TypeError):
            r = Reagent(name='test', concentration=concentration, solubility=solubility)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize('concentration, solubility', [[-1.0, 2.0], [2, 0]])
    def test_init_fail_value_error(self, concentration, solubility):
        with pytest.raises(ValueError):
            r = Reagent(name='test', concentration=concentration, solubility=solubility)

    @pytest.mark.xfail(raises=ValueError)
    def test_init_fail_concentration_greater_than_solubility(self):
        with pytest.raises(ValueError):
            r = Reagent(name='test', concentration=2.0, solubility=1.0)


class TestReagentWV:

    def test_init(self):
        r = ReagentWV(name='test', concentration=1.0)
        assert r.unit == '%(w/v)'


class TestReagentVV:

    def test_init(self):
        r = ReagentVV(name='test', concentration=1.0)
        assert r.unit == '%(v/v)'


class TestBuffer:

    def test_init(self):
        b = Buffer(name='test', concentration=1.0, pH=7.5)
        assert b.pH == 7.5
