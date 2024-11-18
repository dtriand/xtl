from wsgiref.validate import validator

import pytest

from dataclasses import dataclass

from xtl.common.annotated_dataclass import AnnotatedDataclass, afield, cfield, pfield, _ifield


class TestFields:

    def test_afield(self):
        # Default value
        field = afield(param1='value1', default=42)
        assert field.metadata == {'param1': 'value1', 'param_type': 'standard'}
        assert field.default == 42

        # Default factory
        field = afield(param1='value1', default_factory=list)
        assert field.metadata == {'param1': 'value1', 'param_type': 'standard'}
        assert field.default_factory == list

        # Multiple metadata values
        field = afield(param1='value1', param2='value2')
        assert field.metadata == {'param1': 'value1', 'param2': 'value2', 'param_type': 'standard'}
        assert field.default is None

    def test_cfield(self):
        field = cfield(param1='value1')
        assert field.metadata == {'param1': 'value1', 'param_type': 'compound'}

    def test_pfield(self):
        field = pfield(param1='value1')
        assert field.metadata == {'param1': 'value1', 'param_type': 'private'}

    def test_ifield(self):
        field = _ifield(param1='value1')
        assert field.metadata == {'param1': 'value1', 'param_type': '__internal'}


class TestAnnotatedDataclass:

    def test_post_init(self):
        @dataclass
        class TestClass(AnnotatedDataclass):
            param1: int = afield(validator={'ge': 0})
            param2: str = afield(validator={'len': 4})
            param3: str = afield(validator={'choice': ['A', 'B', 'C']})
            param4: list[int] = afield(validator={'len': 3})
            param5: dict = afield()
            param6: tuple = afield(validator={'choices': [1, 2, 3, 4]})
            param7: set = afield()
            param8: float = afield()
            param9: complex = afield()
            param10: str = pfield()
            param11: int = _ifield()

        # Valid initialization
        t = TestClass(param1=42, param2='test', param3='A', param4=[1, 2, 3], param5={'a': 1, 'b': 2},
                      param6=(1, 2, 3), param7={1, 2, 3}, param8=3.14, param9=3+4j)
        assert t.param1 == 42
        assert t.param2 == 'test'
        assert t.param3 == 'A'
        assert t.param4 == [1, 2, 3]
        assert t.param5 == {'a': 1, 'b': 2}
        assert t.param6 == (1, 2, 3)
        assert t.param7 == {1, 2, 3}
        assert t.param8 == 3.14
        assert t.param9 == 3+4j

        # Invalid initialization
        with pytest.raises(ValueError):
            TestClass(param1=-42)
        with pytest.raises(ValueError):
            TestClass(param2='testtesttest')
        with pytest.raises(KeyError):
            TestClass(param3='1')
        with pytest.raises(ValueError):
            TestClass(param4=[1, 2, 3, 4])
        with pytest.raises(ValueError):
            TestClass(param4=[1, 2, '3, 4'])
        with pytest.raises(ValueError):
            TestClass(param6=(1, 5))

    def test_get_param(self):
        @dataclass
        class TestClass(AnnotatedDataclass):
            param1: int = afield(alias='p1')
            param2: str = afield(alias='p2')

        t = TestClass(param1=42, param2='test')
        assert t._get_param('param1').name == 'param1'
        assert t._get_param('p1') is None
        assert t._get_param('param2').name == 'param2'
        assert t._get_param('p2') is None

        assert t._get_param_from_alias('p1').name == 'param1'
        assert t._get_param_from_alias('param1').name == 'param1'
        assert t._get_param_from_alias('p2').name == 'param2'
        assert t._get_param_from_alias('param2').name == 'param2'

    def test_get_param_default_value(self):
        @dataclass
        class TestClass(AnnotatedDataclass):
            param1: int = afield(default=42)
            param2: list[str] = afield(default_factory=list)
            param3: dict = afield(default_factory=lambda: {'a': 1, 'b': 2})

        t = TestClass()
        assert t._get_param_default_value(t._get_param('param1')) == 42
        assert t._get_param_default_value(t._get_param('param2')) == []
        assert t._get_param_default_value(t._get_param('param3')) == {'a': 1, 'b': 2}