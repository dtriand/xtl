from enum import auto

import pytest

from xtl_core.system.compatibility.py310 import StrEnum


class Colour(StrEnum):
    RED = 'red'
    LIGHT_BLUE = auto()


class TestStrEnum:
    # The standard library's from Python 3.11, a stand-in before: both must behave alike

    def test_members_are_strings(self):
        assert isinstance(Colour.RED, str)
        assert Colour.RED == 'red'
        assert Colour('red') is Colour.RED

    @pytest.mark.parametrize('as_text', [str, '{}'.format, lambda member: f'{member}', lambda member: f'{member:>5}'])
    def test_text_is_the_value(self, as_text):
        assert as_text(Colour.RED).strip() == 'red'

    def test_auto_is_the_lowercase_name(self):
        assert Colour.LIGHT_BLUE == 'light_blue'

    def test_repr_names_the_member(self):
        assert repr(Colour.RED) == "<Colour.RED: 'red'>"
