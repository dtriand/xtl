import xtl.GSAS2.GSAS2Interface as GI
from xtl.exceptions import InvalidArgument


class PhaseMixture:

    def __init__(self, phases=[], weight_ratios=[]):
        self._phases = []
        self._ratios = []
        self._ratios_raw = []

        # Validate input
        if len(phases) != len(weight_ratios):
            raise InvalidArgument(message='Phases and weight_ratios must be equal in length.')
        for i, phase in enumerate(phases):
            if not isinstance(phase, GI.G2sc.G2Phase):
                raise InvalidArgument(message=f'Item {i} in phase list is not type G2Phase.')
        for i, ratio in enumerate(weight_ratios):
            if not isinstance(ratio, (float, int)):
                raise InvalidArgument(message=f'Item {i} in weight list is not a number.')
            if ratio < 0:
                raise InvalidArgument(message=f'Item {i} in weight list must be positive number.')

        # Append entries
        for phase, ratio in zip(phases, weight_ratios):
            self._phases += [phase]
            self._ratios_raw += [float(ratio)]
        self._normalize_weight_ratios()

    def add_phase(self, phase, weight_ratio):
        pass

    def remove_phase(self, phase):
        pass

    @property
    def contents(self):
        return [
            {
                'phase': phase.name,
                'weight_ratio': ratio,
                'G2Phase': phase
            }
            for phase, ratio in zip(self._phases, self._ratios)
        ]

    def _normalize_weight_ratios(self):
        ratios_sum = sum(self._ratios_raw)
        for ratio_raw in self._ratios_raw:
            self._ratios += [round(ratio_raw/ratios_sum, 4)]
