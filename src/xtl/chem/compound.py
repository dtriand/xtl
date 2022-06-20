import hashlib
import random
import string
from pathlib import Path

from xtl.units import smart_units, Q_
from xtl.exceptions import InvalidArgument


class Compound:

    COMPOUND_TYPES = ['simple', 'protein', 'w/v']

    def __init__(self, name: str = None, molar_mass: int or float = 0, formula: str = None, type: str = 'simple'):
        """
        Class for representing a chemical compound.

        :param str name: Compound's name
        :param int or float molar_mass: Compound's molar mass in g/mol
        :param str formula: Compound's chemical formula
        :param str type: One of the following: simple, protein, w/v
        """
        if isinstance(name, str):
            self._name = name
        else:
            # Generate a placeholder name, e.g. Compound_5HvmOM (~38 billion combinations)
            self._name = 'Compound_' + ''.join(random.choice(string.ascii_letters + string.digits, k=6))

        self._molar_mass = smart_units(molar_mass, Q_('g/mol'))
        self._formula = formula

        if type not in self.COMPOUND_TYPES:
            raise InvalidArgument(raiser='type', message=f'{type}. Must be one of: {", ".join(self.COMPOUND_TYPES)}')
        self._type = type

        flat_name = self._name if 'Compound_' in self._name and len(self._name) == 15 else self._name.lower()
        self.id = hashlib.md5(bytes(flat_name, 'utf-8')).hexdigest()

    def __repr__(self):
        return f'<{self.__class__.__module__}.{self.__class__.__name__} ({self._name}, {self._molar_mass:0.2f})>'

    @property
    def name(self):
        return self._name

    @property
    def molar_mass(self):
        return self._molar_mass

    def get_mass(self, moles):
        """
        Returns the mass equivalent of the specified moles

        :param int or float or str or pint.Quantity moles: Number of compound moles
        :return: Mass equivalence of ``moles``
        :rtype: pint.Quantity
        """

        n = smart_units(moles, Q_('mol'))
        # n = m / Mr => m = n * Mr
        return n * self._molar_mass


class CompoundLibrary:

    def __init__(self):
        """
        Class for holding a collection of Compound instances.
        """
        self._data = {}

    def add_compound(self, compound: Compound):
        if compound.name not in self._data:
            self._data[compound.name] = compound

    def __getitem__(self, item):
        if item in self._data:
            return self._data[item]
        else:
            print(f'No compound named \'{item}\' in library.')
            return None

    def export_to_file(self, filename: str or Path, overwrite: bool = False):
        fp = Path(filename)
        fp.suffix = '.csv'
        if not fp.exists() or overwrite:
            compounds = dict(sorted(self._data.items()))
            data = '\n'.join([','.join([compound.name, compound.formula, compound.molar_mass])
                              for compound in compounds.values()])
            fp.write_text(data, 'utf-8')
        else:
            print(f'File {fp} already exists. Skipping export...')

    def import_from_file(self, filename: str or Path):
        fp = Path(filename)
        if fp.exists():
            data = fp.read_text('utf-8')
            for i, line in enumerate(data.split('\n')):
                contents = line.split(',')
                if len(contents) == 3:
                    name, formula, molar_mass = contents
                    compound = Compound(name=name, formula=formula, molar_mass=molar_mass)
                    self.add_compound(compound)
                else:
                    print(f'Invalid content at line {i}. Skipping...')
        else:
            raise FileNotFoundError(fp)

    @classmethod
    def from_file(cls, filename: str or Path):
        lib = cls()
        lib.import_from_file(filename)
        return lib

