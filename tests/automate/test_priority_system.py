import pytest

from xtl.automate.priority_system import DefaultPrioritySystem, NicePrioritySystem


class TestDefaultPrioritySystem:

    def test_system_type(self):
        s = DefaultPrioritySystem()
        assert s.system_type is None

    def test_prepare_command(self):
        s = DefaultPrioritySystem()
        assert s.prepare_command('command') == 'command'

        with pytest.raises(ValueError):
            s.prepare_command(1)


class TestNicePrioritySystem:

    def test_system_type(self):
        s = NicePrioritySystem()
        assert s.system_type == 'nice'

    def test_nice_level(self):
        s = NicePrioritySystem()
        assert s.nice_level == 10

        s.nice_level = 5
        assert s.nice_level == 5

        with pytest.raises(ValueError):
            s.nice_level = '5'

    def test_prepare_command(self):
        s = NicePrioritySystem()
        assert s.prepare_command('command') == 'nice -n 10 command'

        s.nice_level = 5
        assert s.prepare_command('command') == 'nice -n 5 command'

        with pytest.raises(ValueError):
            s.prepare_command(1)
