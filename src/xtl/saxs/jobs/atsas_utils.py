from enum import Enum
from pathlib import Path

from pydantic import PrivateAttr, model_validator

from xtl.common.options import Option, Options
from xtl.common.compatibility import PY310_OR_LESS

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...

    from typing_extensions import Self
else:
    from enum import StrEnum
    from typing import Self


class ATSASOptions(Options):
    """
    Base class for ATSAS job options.
    """
    _executable: str = PrivateAttr()

    @property
    def executable(self) -> str:
        return self._executable

    def get_kwargs(self) -> list[str]:
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


class DatopOperator(StrEnum):

    ADD = 'ADD'
    SUBTRACT = 'SUB'
    MULTIPLY = 'MUL'
    DIVIDE = 'DIV'
    NORMALIZE = 'NORM'


class DatopOptions(ATSASOptions):

    _executable: str = PrivateAttr(default='datop')

    operator: DatopOperator = \
        Option(
            desc='Arithmetic operator to apply to dataset'
        )

    value: float | None = \
        Option(
            default=None,
            desc='Numerical value to use with operator'
        )

    dataset: Path | None = \
        Option(
            default=None,
            desc='Path to dataset to use with operator'
        )

    output: Path | None = \
        Option(
            default=None,
            desc='Path to output file'
        )

    @model_validator(mode='after')
    def _mutually_exclusive_options(self) -> Self:
        if self.value is not None and self.dataset is not None:
            raise ValueError('Cannot specify both `dataset` and `value`')
        elif self.value is None and self.dataset is None:
            raise ValueError('Must specify one of: `dataset`, `value`')
        return self