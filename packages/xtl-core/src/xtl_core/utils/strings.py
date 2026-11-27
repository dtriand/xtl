"""
Helpers for strings.
"""


def shorten(text: str, limit: int = 200) -> str:
    """
    Cut a string to ``limit`` characters, ending with ``...`` if it was cut. For values
    in error messages (e.g. ``shorten(repr(value))``), so that a huge input doesn't end
    up in full in messages and logs.
    """
    return text if len(text) <= limit else f'{text[: limit - 3]}...'
