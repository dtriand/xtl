

class ComputeSite:

    def load_modules(self, modules):
        raise NotImplementedError()

    def purge_modules(self):
        raise NotImplementedError()


class LocalSite(ComputeSite):

    def load_modules(self, modules):
        return ''

    def purge_modules(self):
        return ''


class BiotixHPC(ComputeSite):

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

