import logging

import pytest

from xtl_core.utils.imports import import_object, import_path
from xtl_core.utils.mappings import deepmerge


class TestImports:
    def test_round_trip(self):
        assert import_path(logging.FileHandler) == 'logging.FileHandler'
        assert import_object('logging.FileHandler') is logging.FileHandler
        assert import_object(import_path(deepmerge)) is deepmerge

    @pytest.mark.parametrize('path', ['logging', '', 'logging.NoSuchHandler', 'no_such_module.X'])
    def test_errors(self, path):
        with pytest.raises(ImportError):
            import_object(path)
