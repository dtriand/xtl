import abc
from pathlib import Path
from typing import Optional


from xtl_core.options import Option
from xtl_core.options.validators import cast_as_temp_dir_if_none
from xtl.jobs.jobs import BatchJob
from xtl.jobs.config import BatchJobConfig
from xtl.jobs.shells import Shell
from xtl.scattering.jobs.atsas_utils import DatcmpOptions, DatopOptions


class ATSASBatchJobConfig(BatchJobConfig, abc.ABC):
    """
    Base configuration for ATSAS batch jobs.
    """

    job_directory: Optional[Path] = Option(
        default_factory=lambda: cast_as_temp_dir_if_none(None, prefix='xtl_atsas_'),
        desc='Directory for job execution and results',
        cast_as=lambda x: cast_as_temp_dir_if_none(x, prefix='xtl_atsas_'),
    )
    filename: str = 'atsas'
    name: Optional[str] = 'xtl.ATSASBatchJob'
    description: Optional[str] = 'ATSAS batch job'
    templates: dict[Shell, str] = Option(
        default_factory=lambda: {
            Shell.BASH: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
            '__ATSAS_EXEC__ __INPUT_FILES__ __ATSAS_KWARGS____XTL_NL__',
            Shell.CMD: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
            '__ATSAS_EXEC__ __INPUT_FILES__ __ATSAS_KWARGS____XTL_NL__',
            Shell.POWERSHELL: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
            '__ATSAS_EXEC__ __INPUT_FILES__ __ATSAS_KWARGS____XTL_NL__',
        },
        desc='Templates for the content of the batch file for different shells',
    )

    def get_context(self) -> dict:
        context = super().get_context()
        if hasattr(self, 'options') and hasattr(self.options, '_executable'):
            context |= {'ATSAS_EXEC': self.options._executable}
        return context


class DatcmpBatchJobConfig(ATSASBatchJobConfig):
    filename: str = 'datcmp'
    name: Optional[str] = 'xtl.DatcmpBatchJobConfig'
    description: Optional[str] = 'DATCMP batch job'

    input: list[Path] = Option(desc='List of input data files for `datcmp`')
    options: DatcmpOptions = Option(default_factory=DatcmpOptions, desc='Options for `datcmp`')

    def get_context(self):
        context = super().get_context()
        context['ATSAS_KWARGS'] = ' '.join(self.options.get_kwargs())
        # A list of paths, each quoted for the shell by the batch file
        context['INPUT_FILES'] = list(self.input)
        return context


class DatcmpBatchJob(BatchJob[DatcmpBatchJobConfig]):
    """
    Job to compare SAXS datasets using ATSAS `datcmp`.
    """

    _dependencies = {'atsas'}


class DatopBatchJobConfig(ATSASBatchJobConfig):
    filename: str = 'datop'
    name: Optional[str] = 'xtl.DatopBatchJobConfig'
    description: Optional[str] = 'DATOP batch job'
    templates: dict[Shell, str] = Option(
        default_factory=lambda: {
            Shell.BASH: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
            '__ATSAS_EXEC__ __OPERATOR__ __INPUT_FILE__ __OP_VALUE__ __ATSAS_KWARGS____XTL_NL__',
            Shell.CMD: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
            '__ATSAS_EXEC__ __OPERATOR__ __INPUT_FILE__ __OP_VALUE__ __ATSAS_KWARGS____XTL_NL__',
            Shell.POWERSHELL: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
            '__ATSAS_EXEC__ __OPERATOR__ __INPUT_FILE__ __OP_VALUE__ __ATSAS_KWARGS____XTL_NL__',
        },
        desc='Templates for the content of the batch file for different shells',
    )

    input: Path = Option(desc='Path to input file for `datop`')
    options: DatopOptions = Option(desc='Options for `datop`')

    def get_context(self):
        context = super().get_context()
        # Paths are quoted for the shell by the batch file: the output as an option and a path of its own
        context |= {
            'INPUT_FILE': self.input,
            'OPERATOR': self.options.operator,
            'OP_VALUE': self.options.value if self.options.value is not None else self.options.dataset,
            'ATSAS_KWARGS': ['--output', self.options.output] if self.options.output is not None else '',
        }
        return context


class DatopBatchJob(BatchJob[DatopBatchJobConfig]):
    """
    Job to compare SAXS datasets using ATSAS `datop`.
    """

    _dependencies = {'atsas'}
