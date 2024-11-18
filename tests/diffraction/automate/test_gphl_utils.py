import pytest

from dataclasses import dataclass
from pathlib import Path

from xtl.diffraction.automate.gphl_utils import GPhLConfig
from xtl.common.annotated_dataclass import afield, cfield, pfield, _ifield


class TestGPhLConfig:

    def test_alias(self):
        @dataclass
        class TestConfig(GPhLConfig):
            a: int = afield(alias='A')
            b: int = afield(default=5)

        config = TestConfig(a='4')
        assert config.a == 4
        assert config.b == 5
        assert config.get_param_value('a') == {'A': 4}
        assert config.get_param_value('b') == {'b': 5}

    def test_alias_fstring(self):
        @dataclass
        class TestConfig(GPhLConfig):
            a: str = afield(alias_fstring='{a} {b}', alias_fstring_keys=['a', 'b'])
            b: str = afield()

        config = TestConfig(a='Hello', b='World')
        assert config.a == 'Hello'
        assert config.get_param_value('a') == {'a': '"Hello World"'}

    def test_type_formatting(self):
        @dataclass
        class TestConfig(GPhLConfig):
            a: bool = afield()
            b: str = afield()
            c: float = afield(formatter=lambda x: f'{x:.4f}')
            d: list[int] = afield(formatter=lambda x: " ".join(map(str, x)))

        config = TestConfig(a=True, b='Hello', c=3, d=[1, 2, 3])
        assert config.get_param_value('a') == {'a': 'yes'}
        assert config.get_param_value('b') == {'b': 'Hello'}
        assert config.get_param_value('c') == {'c': '3.0000'}
        assert config.get_param_value('d') == {'d': '"1 2 3"'}

    def test_compound_fields(self):
        @dataclass
        class TestConfig(GPhLConfig):
            a: str = pfield()
            b: int = pfield(formatter=lambda x: x * 2)
            c: int = cfield(members=['a', 'b'])
            d: int = cfield(members=['a', 'b'],
                            formatter=lambda x: {f'_{k}': v for k, v in x.items()})

        config = TestConfig(a=1, b=2)
        assert config.get_param_value('c') == {'a': '1', 'b': 4}

        config = TestConfig(b=2)
        assert config.get_param_value('d') == {'_a': None, '_b': 4}

    def test_groups(self):
        @dataclass
        class TestConfig(GPhLConfig):
            a: int = afield(group='Group1')
            b: int = afield(group='Group1', formatter=lambda x: x * 2)
            c: int = afield(group='Group2')
            d: int = afield(group='Group2', formatter=lambda x: x * 2)

        config = TestConfig(a=1, b=2, c=3, d=4)
        assert config.get_group('Group1') == {'a': 1, 'b': 4}
        assert config.get_group('Group2') == {'c': 3, 'd': 8}

    def test_all_params(self):
        @dataclass
        class TestConfig(GPhLConfig):
            a: int = afield(group='Group1', default=1)
            b: int = afield(group='Group1', formatter=lambda x: x * 2)
            c: int = afield(group='Group2', default=3)
            d: int = afield(group='Group2', formatter=lambda x: x * 2)
            _groups: dict = _ifield(default_factory=lambda: {'Group1': 'Comment1', 'Group2': 'Comment2'})

        config = TestConfig(b=2, d=4)
        assert config.get_all_params(grouped=True) == {'Group1': {'comment': 'Comment1', 'params': {'a': 1, 'b': 4}},
                                                       'Group2': {'comment': 'Comment2', 'params': {'c': 3, 'd': 8}}}
        assert config.get_all_params(grouped=True, modified_only=True) == {'Group1': {'comment': 'Comment1',
                                                                                      'params': {'b': 4}},
                                                                           'Group2': {'comment': 'Comment2',
                                                                                      'params': {'d': 8}}}
        assert config.get_all_params(grouped=False) == {'a': 1, 'b': 4, 'c': 3, 'd': 8}
        assert config.get_all_params(grouped=False, modified_only=True) == {'b': 4, 'd': 8}

        # Missing _groups attribute
        @dataclass
        class TestConfig(GPhLConfig):
            a: int = afield(group='Group1', default=1)
            b: int = afield(group='Group1', formatter=lambda x: x * 2)
            c: int = afield(group='Group2', default=3)
            d: int = afield(group='Group2', formatter=lambda x: x * 2)

        config = TestConfig(b=2, d=4)
        assert config.get_all_params(grouped=True) == {'a': 1, 'b': 4, 'c': 3, 'd': 8}
        assert config.get_all_params(grouped=True, modified_only=True) == {'b': 4, 'd': 8}

    def test_format_value(self):
        assert GPhLConfig._format_value(value=True) == 'yes'
        assert GPhLConfig._format_value(value=False) == 'no'
        assert GPhLConfig._format_value(value='Hello') == 'Hello'
        assert GPhLConfig._format_value(value='Hello World!') == '"Hello World!"'
        assert GPhLConfig._format_value(value=Path('path/to/file')) == '"path/to/file"'
        assert GPhLConfig._format_value(value=[1, 2, 3], formatter=lambda x: ' '.join(map(str, x))) == '"1 2 3"'