from pathlib import Path

from xtl.common.options import Option
from xtl.jobs import SteppedJob, SteppedJobConfig, StepSpec
from xtl.files.jobs.directories import CreateDirectoryJob, CreateDirectoryJobConfig
from xtl.scattering.jobs.atsas import DatopBatchJob, DatopBatchJobConfig
from xtl.scattering.jobs.atsas_utils import DatopOptions, DatopOperator


class SAXSSubtractJobConfig(SteppedJobConfig):
    input: Path = \
        Option(
            ...,
            desc='Path to input SAXS dataset',
            path_exists=True,
        )

    modifier: float | Path = \
        Option(
            ...,
            desc='Numerical value or path to dataset to subtract from input dataset',
        )

    output: Path | None = \
        Option(
            default=None,
            desc='Path to output SAXS dataset',
        )


class SAXSSubtractJob(SteppedJob[SAXSSubtractJobConfig]):
    """
    Job to subtract one SAXS dataset from another using ATSAS `datop`.
    """
    _steps = (
        StepSpec(
            name='create_dir',
            desc='Create directory for output dataset',
            job_cls=CreateDirectoryJob,
            config_cls=CreateDirectoryJobConfig,
            condition=lambda ctx: ctx.parent.config.output is not None,
            dynamic_defaults=lambda ctx: {
                'directory': ctx.parent.config.output.parent,
            }
        ),
        StepSpec(
            name='datop_batch',
            desc='Run datop batch job to subtract datasets',
            job_cls=DatopBatchJob,
            config_cls=DatopBatchJobConfig,
            defaults={
                'options': {
                    'operator': DatopOperator.SUBTRACT
                }
            },
            dynamic_defaults=lambda ctx: {
                'input': ctx.parent.config.input,
                'options': {
                    'value': ctx.parent.config.modifier
                        if isinstance(ctx.parent.config.modifier, (int, float)) else None,
                    'dataset': ctx.parent.config.modifier
                        if isinstance(ctx.parent.config.modifier, Path) else None,
                    'output': ctx.parent.config.output
                }
            }
        ),

    )