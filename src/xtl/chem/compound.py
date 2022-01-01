import hashlib
import random
import string

from xtl.units import smart_units, Q_
from xtl.exceptions import InvalidArgument


class Compound:

    COMPOUND_TYPES = ['simple', 'protein', 'w/v']

    def __init__(self, name: str = None, molar_mass=0, formula: str = None, type: str = 'simple'):
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

