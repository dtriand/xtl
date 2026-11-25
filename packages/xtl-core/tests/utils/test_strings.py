from hypothesis import given, strategies as st
import pytest

from xtl_core.utils.strings import shorten


class TestShorten:
    @pytest.mark.parametrize('text', ['', 'abc', 'a' * 200], ids=['empty', 'short', 'at-limit'])
    def test_unchanged(self, text):
        assert shorten(text) == text

    def test_cut(self):
        assert shorten('a' * 201) == 'a' * 197 + '...'

    def test_limit(self):
        assert shorten('abcdefgh', 5) == 'ab...'

    @given(text=st.text(), limit=st.integers(3, 50))
    def test_length_and_prefix(self, text, limit):
        result = shorten(text, limit)
        assert len(result) <= limit
        assert result == text or (result.endswith('...') and text.startswith(result[:-3]))
