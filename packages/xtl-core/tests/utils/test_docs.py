from xtl_core.utils.docs import extend_docstring


def _parent():
    """
    Do something.

    :param a: The a.
    :return: Something.
    :raises ValueError: Sometimes.
    """


def _description_only():
    """
    Only a description.
    """


def _no_return():
    """
    Do something.

    :param a: The a.
    """


def _no_docstring():
    pass


class TestExtendDocstring:
    def extend(self, parent):
        @extend_docstring(parent, note='The note.', params={'b': 'The b.'})
        def func():
            pass

        return func

    def test_returns_func(self):
        def func():
            pass

        assert extend_docstring(_parent, note='n', params={})(func) is func

    def test_full(self):
        assert self.extend(_parent).__doc__ == (
            'Do something.\n'
            '\n'
            'The note.\n'
            '\n'
            ':param a: The a.\n'
            ':param b: The b.\n'
            ':return: Something.\n'
            ':raises ValueError: Sometimes.'
        )

    def test_description_only(self):
        """The note stays a paragraph of its own."""
        assert self.extend(_description_only).__doc__ == 'Only a description.\n\nThe note.\n\n:param b: The b.'

    def test_no_return(self):
        assert self.extend(_no_return).__doc__.endswith(':param a: The a.\n:param b: The b.')

    def test_no_docstring(self):
        assert self.extend(_no_docstring).__doc__ == 'The note.\n\n:param b: The b.'

    def test_parent_unchanged(self):
        before = _parent.__doc__
        self.extend(_parent)
        assert _parent.__doc__ == before
