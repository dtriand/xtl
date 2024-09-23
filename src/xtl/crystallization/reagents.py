from dataclasses import dataclass

from .applicators import _ReagentApplicator


@dataclass
class _Reagent:
    name: str
    concentration: float
    unit: str
    fmt_str: str
    solubility: float
    applicator: _ReagentApplicator = None

    def __init__(self, name: str, concentration: float, solubility: float = None, fmt_str: str = None):
        self.name = name

        if not isinstance(concentration, (int, float)):
            raise TypeError(f'Concentration must be an integer or float, not {type(concentration)}')
        if concentration <= 0:
            raise ValueError('Concentration must be positive')
        self.concentration = float(concentration)

        if solubility is not None:
            if not isinstance(solubility, (int, float)):
                raise TypeError(f'Solubility must be an integer or float, not {type(solubility)}')
            if solubility <= 0:
                raise ValueError('Solubility must be positive')
            self.solubility = float(solubility)
        else:
            self.solubility = None

        if self.solubility is not None and self.concentration > self.solubility:
            raise ValueError('Concentration must be less than or equal to solubility')

        self.unit = 'M'
        self.fmt_str = fmt_str
        self._location = list()
        self._repr_keywords = ['name', 'concentration', 'unit', 'solubility']

    def __repr__(self):
        keywords = [f'{key}={self.__getattribute__(key)}' for key in self._repr_keywords]
        return f'{type(self).__name__}({", ".join(keywords)})'

    def to_dict(self):
        return {
            'name': self.name,
            'type': type(self).__name__,
            'concentration': self.concentration,
            'unit': self.unit,
            'solubility': self.solubility,
            'applicator': self.applicator.to_dict() if self.applicator is not None else None,
            'location': self._location
        }


@dataclass
class Reagent(_Reagent):

    def __init__(self, name: str, concentration: float, solubility: float = None, fmt_str: str = None):
        super().__init__(name=name, concentration=concentration, solubility=solubility, fmt_str=fmt_str)

    def __repr__(self):
        return super().__repr__()

@dataclass
class ReagentWV(_Reagent):

    def __init__(self, name: str, concentration: float, solubility: float = None, fmt_str: str = None):
        super().__init__(name=name, concentration=concentration, solubility=solubility, fmt_str=fmt_str)
        self.unit = '%(w/v)'

    def __repr__(self):
        return super().__repr__()


@dataclass
class ReagentVV(_Reagent):

    def __init__(self, name: str, concentration: float, solubility: float = None, fmt_str: str = None):
        super().__init__(name=name, concentration=concentration, solubility=solubility, fmt_str=fmt_str)
        self.unit = '%(v/v)'

    def __repr__(self):
        return super().__repr__()


@dataclass
class ReagentBuffer(_Reagent):

    def __init__(self, name: str, concentration: float, pH: float, solubility: float = None, fmt_str: str = None):
        super().__init__(name=name, concentration=concentration, solubility=solubility, fmt_str=fmt_str)

        if not isinstance(pH, (int, float)):
            raise TypeError(f'pH must be an integer or float, not {type(pH)}')
        if (pH < 0) or (pH > 14):
            raise ValueError('pH must be between 0 and 14')
        self.pH = float(pH)
        self.pH_applicator: _ReagentApplicator = None

        self.unit = 'M'
        self._repr_keywords += ['pH']

    def __repr__(self):
        return super().__repr__()

    def to_dict(self):
        d = super().to_dict()
        d['pH'] = self.pH
        d['pH_applicator'] = self.pH_applicator.to_dict() if self.pH_applicator is not None else None
        return d


Buffer = ReagentBuffer
