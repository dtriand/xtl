from __future__ import annotations

from typing import Any, Optional, Callable

from xtl.common.options import Option, Options
from xtl.jobs.jobs import Job, BatchJob
from xtl.jobs.config import JobConfig, BatchJobConfig
from xtl.jobs.results import JobResults


__all__ = ['StepSpec', 'JobContext']


class StepSpec(Options):
    model_config = Options.model_config | {'frozen': True}

    name: str = \
        Option(
            ...,
            desc='Name of the step'
        )
    desc: Optional[str] = \
        Option(
            default=None,
            desc='Optional description of the step'
        )
    job_cls: type[Job | BatchJob] = \
        Option(
            ...,
            desc='Class of the job to execute for this step. Must be a subclass of Job.'
        )
    config_cls: type[JobConfig | BatchJobConfig] = \
        Option(
            ...,
            desc='Class of the configuration for this step. Must be a subclass of JobConfig.'
        )
    defaults: dict[str, Any] = \
        Option(
            default_factory=dict,
            desc='Default values for the step configuration. Keys should match the fields of the config class.'
        )
    dynamic_defaults: Callable[[JobContext], dict[str, Any]] = \
        Option(
            default_factory=lambda: lambda ctx: {},
            desc='A function that takes the JobContext and returns a dictionary of dynamic default values for the '
                 'step configuration. This can be used to set defaults based on the results of previous steps.'
        )
    post_processor: Optional[Callable[[JobResults | None, JobContext], dict[str, Any]]] = \
        Option(
            default=None,
            desc='An optional function that takes the JobContext and the result of the step, and can modify the '
                 'results before being stored to the JobContext.'
        )

    def __repr__(self):
        return f'{StepSpec.__name__}<{self.job_cls.__name__}[{self.config_cls.__name__}]>(name={self.name!r}, ' \
               f'defaults={self.defaults!r}, dynamic_defaults={self.dynamic_defaults!r}, ' \
               f'post_processor={self.post_processor!r})'

    def __rich_repr__(self):
        yield f'<{self.job_cls.__name__}[{self.config_cls.__name__}]>'
        yield 'name', self.name
        yield 'desc', self.desc
        yield 'defaults', self.defaults
        yield 'dynamic_defaults', self.dynamic_defaults
        yield 'post_processor', self.post_processor


class JobContext(Options):

    parent: Job = \
        Option(
            ...,
            desc='The parent job that is executing the steps. This can be used to access shared resources or state.'
        )
    step: Optional[Job | BatchJob] = \
        Option(
            default=None,
            desc='The current step job being executed. This will be set when a step is running, and can be used to '
                 'access the step configuration or other attributes.'
        )
    results: dict[str, Any] = \
        Option(
            default_factory=dict,
            desc='A dictionary to store results from each step. Steps can read and write to this dictionary to '
                 'share data.'
        )
    data: dict[str, Any] = \
        Option(
            default_factory=dict,
            desc='A dictionary to store any additional data or state that may be shared across steps.'
        )
