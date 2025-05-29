class _Undefined:
    """
    Sentinel class for undefined values.

    Always use ``==`` operator to compare with XTLUndefined, not `is`.
    """

    # To make this a singleton
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_Undefined, cls).__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'XTLUndefined'

    def __str__(self):
        return 'XTLUndefined'

    def __bool__(self):
        return False

    # Equality comparison
    def __eq__(self, other):
        if isinstance(other, _Undefined):
            return True
        return False



# Global singleton instance
XTLUndefined = _Undefined()
"""A singleton instance for undefined values."""