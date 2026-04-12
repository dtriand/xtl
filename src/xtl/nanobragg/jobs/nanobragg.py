from pathlib import Path
from typing import Optional, TYPE_CHECKING

from xtl.common.options import Option
from xtl.common.validators import cast_as_temp_dir_if_none
from xtl.exceptions.base import StderrError
from xtl.jobs.jobs import Job, BatchJob, SteppedJob
from xtl.jobs.steps import StepSpec
from xtl.jobs.config import JobConfig, BatchJobConfig, SteppedJobConfig
from xtl.jobs.shells import Shell

from xtl.nanobragg.config import NanoBraggOptions


class NanoBraggBatchJobConfig(BatchJobConfig):
    """
    Configuration for a nanoBragg batch job.
    """
    job_directory: Optional[Path] = Option(
        default_factory=lambda: cast_as_temp_dir_if_none(None, prefix='xtl_nanobragg_'),
        desc='Directory for job execution and results',
        cast_as=lambda x: cast_as_temp_dir_if_none(x, prefix='xtl_nanobragg_'),
    )
    filename: str = 'nanobragg'
    name: Optional[str] = 'xtl.NanoBraggBatchJob'
    description: Optional[str] = 'nanoBragg batch job'
    dependencies: set[str] = \
        Option(
            default_factory=lambda: {'easybragg'},
            desc='List of dependencies required for this batch job'
        )
    templates: dict[Shell, str] = \
        Option(
            default_factory=lambda: {
                Shell.BASH:
                    '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
                    'easyBragg.python __NANOBRAGG_SCRIPT__ --config=__NANOBRAGG_CONFIG__ '
                        '--output=__NANOBRAGG_OUTPUT_DIR__ __NANOBRAGG_EXTRA_ARGS____XTL_NL__',
            },
            desc='Templates for the content of the batch file for different shells'
        )
    input: Path = \
        Option(
            desc='Path to input options JSON file',
        )
    use_gpu: bool = \
        Option(
            default=False,
            desc='Use GPU acceleration for nanoBragg'
        )
    debug: bool = \
        Option(
            default=False,
            desc='Enable debug mode for nanoBragg'
        )

    def get_extra_args(self) -> str:
        """
        Get extra command-line arguments for nanoBragg based on the configuration.
        """
        extra = []
        if self.use_gpu:
            extra.append('--gpu')
        if self.debug:
            extra.append('--debug')
        return ' '.join(extra).rstrip(' ')

    def get_context(self):
        context = super().get_context()
        context['NANOBRAGG_CONFIG'] = self.input
        context['NANOBRAGG_OUTPUT_DIR'] = self.job_directory
        context['NANOBRAGG_EXTRA_ARGS'] = self.get_extra_args()
        return context


class NanoBraggBatchJob(BatchJob[NanoBraggBatchJobConfig]):

    _keep_files = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_context |= {
            'NANOBRAGG_SCRIPT': str(Path(__file__).parent / 'scripts' / 'nanobragg.py')
        }


class NanoBraggJobConfig(SteppedJobConfig):
    ...


class NanoBraggJob(SteppedJob[NanoBraggJobConfig]):

    _keep_files = True

    _steps = (
        StepSpec(
            name='nanobragg_batch',
            desc='Run nanoBragg batch job',
            job_cls=NanoBraggBatchJob,
            config_cls=NanoBraggBatchJobConfig,
            post_processor=lambda results, ctx: {
                'cbf': NanoBraggJob.glob_files(ctx.step.config.job_directory, '*.cbf'),
                'npy': NanoBraggJob.glob_files(ctx.step.config.job_directory, '*.npy'),
                'png': NanoBraggJob.glob_files(ctx.step.config.job_directory, '*.png'),
            }
        ),
    )

    @staticmethod
    def glob_files(path: Path, pattern: str) -> list[Path]:
        return sorted(list(Path(path).glob(pattern)))
