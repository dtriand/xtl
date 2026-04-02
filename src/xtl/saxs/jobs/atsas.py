import abc
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import logging

from xtl.common.options import Option
from xtl.common.validators import cast_as_temp_dir_if_none
from xtl.jobs.jobs import BatchJob
from xtl.jobs.config import BatchJobConfig
from xtl.jobs.shells import Shell
from xtl.saxs.jobs.atsas_utils import ATSASOptions, DatcmpOptions


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
    dependencies: set[str] = \
        Option(
            default_factory=lambda: {'atsas'},
            desc='List of dependencies required for this batch job'
        )
    templates: dict[Shell, str] = \
        Option(
            default_factory=lambda: {
                Shell.BASH:
                    '__XTL_COMMENT__ __XTL_DOCSTRING__ __XTL_NL__'
                    '__ATSAS_EXEC__ __INPUT_FILES__ __ATSAS_KWARGS__',
                Shell.CMD:
                    '__XTL_COMMENT__ __XTL_DOCSTRING__ __XTL_NL__'
                    '__ATSAS_EXEC__ __INPUT_FILES__ __ATSAS_KWARGS__',
                Shell.POWERSHELL:
                    '__XTL_COMMENT__ __XTL_DOCSTRING__ __XTL_NL__'
                    '__ATSAS_EXEC__ __INPUT_FILES__ __ATSAS_KWARGS__',
            },
            desc='Templates for the content of the batch file for different shells'
        )


class DatcmpBatchJobConfig(ATSASBatchJobConfig):
    filename: str = 'datcmp'
    name: Optional[str] = 'xtl.DatcmpBatchJobConfig'
    description: Optional[str] = 'DATCMP batch job'

    input: list[Path] = \
        Option(
            desc='List of input data files for `datcmp`'
        )
    options: DatcmpOptions = \
        Option(
            default_factory=DatcmpOptions,
            desc='Options for `datcmp`'
        )

    def get_context(self):
        context = super().get_context()
        context['ATSAS_KWARGS'] = ' '.join(self.options.get_args())
        context['INPUT_FILES'] = ' '.join(self.shell.get().sanitize_value(file) for file in self.input)
        return context


class DatcmpBatchJob(BatchJob[ATSASBatchJobConfig]):
    """
    Job to compare SAXS datasets using ATSAS datcmp.
    """

    def __init__(self, job_id: str | None = None, logger: 'logging.Logger' = None):
        super().__init__(job_id=job_id, logger=logger)
        self._batch_context |= {
            'ATSAS_EXEC': 'datcmp'
        }
