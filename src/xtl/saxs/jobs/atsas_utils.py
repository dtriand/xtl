from enum import Enum

from pydantic import PrivateAttr

from xtl.common.options import Option, Options
from xtl.common.compatibility import PY310_OR_LESS

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class ATSASOptions(Options):
    """
    Base class for ATSAS job options.
    """
    _executable: str = PrivateAttr()

    @property
    def executable(self) -> str:
        return self._executable

    def get_args(self) -> list[str]:
        """
        Returns the command-line arguments for the ATSAS executable.
        """
        args = []
        for key, value in self.to_dict(by_alias=True).items():
            if value is not None:
                args.append(f'--{key}={value}')
        return args


class DatcmpMode(StrEnum):
    PAIRWISE = 'PAIRWISE'
    INDEPENDENT = 'INDEPENDENT'


class DatcmpTest(StrEnum):
    CORMAP = 'CORMAP'
    CHI_SQUARE = 'CHI-SQUARE'
    ANDERSON_DARLING = 'ANDERSON-DARLING'


class DatcmpAdjustment(StrEnum):
    FWER = 'FWER'
    FDR = 'FDR'


class DatcmpOptions(ATSASOptions):
    """
    Configuration for an ATSAS datcmp job.
    """
    _executable: str = PrivateAttr(default='datcmp')

    mode: DatcmpMode | None = \
        Option(
            default='PAIRWISE',
            desc='Comparison mode'
        )
    test: DatcmpTest | None = \
        Option(
            default='CORMAP',
            desc='Test name'
        )
    adjust: DatcmpAdjustment | None = \
        Option(
            default='FWER',
            desc='Adjustment for multiple testing'
        )
    alpha: float | None = \
        Option(
            default=0.01,
            desc='Significance level for clique search'
        )
    format: str | None = \
        Option(
            default='FULL',
            desc='Output format',
            choices={'FULL', 'CSV', None}
        )
