from priority_system import _PrioritySystem, DefaultPrioritySystem, NicePrioritySystem


class ComputeSite:
    _priority_system = DefaultPrioritySystem()

    @property
    def priority_system(self):
        return self._priority_system

    @priority_system.setter
    def priority_system(self, value):
        if not isinstance(value, _PrioritySystem):
            raise TypeError('priority_system must be an instance of _PrioritySystem')
        self._priority_system = value

    def load_modules(self, modules):
        raise NotImplementedError()

    def purge_modules(self):
        raise NotImplementedError()

    def prepare_command(self, command: str):
        return self.priority_system.prepare_command(command)


class LocalSite(ComputeSite):

    def load_modules(self, modules):
        return ''

    def purge_modules(self):
        return ''


class BiotixHPC(ComputeSite):
    _priority_system = NicePrioritySystem(10)

    def load_modules(self, modules: str | list[str]):
        mods = []
        if isinstance(modules, str):
            for mod in modules.split():
                mods.append(mod)
        else:
            for mod in modules:
                mods.append(mod)
        if not mods:
            return ''
        return 'module load ' + ' '.join(mods)

    def purge_modules(self):
        return 'module purge'
