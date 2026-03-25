from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import logging

from xtl.common.options import Option
from xtl.common.validators import cast_as_temp_dir_if_none
from xtl.exceptions.base import StderrError
from xtl.jobs.jobs import Job, BatchJob
from xtl.jobs.config2 import BatchJobConfig
from xtl.jobs.config import JobConfig, JobStepsConfig
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
                    'easyBragg.python __NANOBRAGG_SCRIPT__ --config=__NANOBRAGG_CONFIG__',
            },
            desc='Templates for the content of the batch file for different shells'
        )


class NanoBraggBatchJob(BatchJob[NanoBraggBatchJobConfig]):

    def __init__(self, job_id: str | None = None, logger: 'logging.Logger' = None):
        super().__init__(job_id=job_id, logger=logger)
        self._batch_context |= {
            'NANOBRAGG_SCRIPT': str(Path(__file__).parent / 'scripts' / 'nanobragg.py')
        }


class NanoBraggJobStepsConfig(JobStepsConfig, total=False):
    nanobragg_batch: NanoBraggBatchJobConfig


class NanoBraggJobConfig(JobConfig):
    options: NanoBraggOptions
    steps: NanoBraggJobStepsConfig = \
        Option(
            default_factory=lambda: NanoBraggJobStepsConfig(
                nanobragg_batch=NanoBraggBatchJobConfig()
            ),
            desc='Steps for the nanoBragg job'
        )


class NanoBraggJob(Job[NanoBraggJobConfig]):

    async def _execute(self) -> dict[str, list[Path]]:
        no_steps = len(self.config.steps_list)

        ###############################
        # Step 1: nanoBragg batch job #
        ###############################
        i = 0
        step = self.config.steps_list[i]
        self.logger.info('Executing step %(i)d/%(n)d: %(steps)s', {'i': i + 1, 'n': no_steps, 'steps': step})

        # Prepare batch job
        self.logger.debug(f'Preparing {NanoBraggBatchJob.__name__}')
        batch_config = self.config.steps[step]
        options_json = batch_config.job_directory / 'nanobragg_options.json'
        batch_job = NanoBraggBatchJob.with_config(
            job_id=f'{self.job_id}.{i+1}',
            config=batch_config,
            **{
                'NANOBRAGG_CONFIG': options_json
            }
        )

        # Save options to file
        if options_json.exists():
            raise FileExistsError(f'Options file already exists: {options_json}')
        try:
            self.logger.debug(f'Creating directory for options file: {options_json.parent}')
            options_json.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f'Failed to create directory for options file: {options_json.parent}') from e
        self.logger.debug(f'Saving nanoBragg options to {options_json}')
        self.config.options.to_json(options_json)

        # Run batch job
        self.logger.info('Running nanoBragg batch job in: %(dir)s', {'dir': batch_config.job_directory})
        results = await batch_job.run()
        self.logger.debug(f'Batch job completed')
        if results.error:
            raise RuntimeError(f'nanoBragg batch job failed with error: {results.error}')
        else:
            if stderr := results.data.get('stderr', None):
                raise StderrError('nanoBragg batch job failed', stderr=stderr)

        # Extract files
        self.logger.debug('Collecting results from batch job')
        images = {
            'cbf': list(batch_config.job_directory.glob('image_*.cbf')),
            'npy': list(batch_config.job_directory.glob('image_*.npy')),
            'png': list(batch_config.job_directory.glob('image_*.png')),
        }
        self.logger.debug('Step %(i)d/%(n)d completed: %(step)s', {'i': i + 1, 'n': no_steps, 'step': step})

        self.logger.info('All steps completed')
        return images
