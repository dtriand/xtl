import pytest

from xtl.automate.priority_system import DefaultPrioritySystem, NicePrioritySystem
from xtl.automate.sites import ComputeSite, LocalSite, BiotixHPC



class TestLocalSite:

    def test_priority_system(self):
        cs = LocalSite()
        assert isinstance(cs.priority_system, DefaultPrioritySystem)

        cs.priority_system = NicePrioritySystem(10)
        assert isinstance(cs.priority_system, NicePrioritySystem)

    def test_load_modules(self):
        cs = LocalSite()
        assert cs.load_modules('module1') == ''

    def test_purge_modules(self):
        cs = LocalSite()
        assert cs.purge_modules() == ''

    def test_prepare_command(self):
        cs = LocalSite()
        assert cs.prepare_command('command') == 'command'


class TestBiotixHPC:

    def test_priority_system(self):
        cs = BiotixHPC()
        assert isinstance(cs.priority_system, NicePrioritySystem)

        cs.priority_system.nice_level = 5
        assert cs.priority_system.nice_level == 5

        cs.priority_system = DefaultPrioritySystem()
        assert isinstance(cs.priority_system, DefaultPrioritySystem)

    def test_load_modules(self):
        cs = BiotixHPC()
        assert cs.load_modules('module1') == 'module load module1'
        assert cs.load_modules('module1 module2') == 'module load module1 module2'
        assert cs.load_modules(['module1', 'module2']) == 'module load module1 module2'
        with pytest.raises(ValueError):
            cs.load_modules(1)
        with pytest.raises(ValueError):
            cs.load_modules([1])

    def test_purge_modules(self):
        cs = BiotixHPC()
        assert cs.purge_modules() == 'module purge'

    def test_prepare_command(self):
        cs = BiotixHPC()
        assert cs.prepare_command('command') == 'nice -n 10 command'

        cs.priority_system = NicePrioritySystem(5)
        assert cs.prepare_command('command') == 'nice -n 5 command'