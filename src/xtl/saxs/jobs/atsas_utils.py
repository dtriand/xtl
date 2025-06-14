from pydantic import PrivateAttr

from xtl.common.options import Option, Options


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


class DatcmpOptions(ATSASOptions):
    """
    Configuration for an ATSAS datcmp job.
    """
    _executable: str = PrivateAttr(default='datcmp')

    mode: str | None = \
        Option(
            default='PAIRWISE',
            desc='Comparison mode',
            choices={'PAIRWISE', 'INDEPENDENT', None}
        )
    test: str | None = \
        Option(
            default='CORMAP',
            desc='Test name',
            choices={'CORMAP', 'CHI-SQUARE', 'ANDERSON-DARLING', None}
        )
    adjust: str | None = \
        Option(
            default='FWER',
            desc='Adjustment for multiple testing',
            choices={'FWER', 'FDR', None}
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
