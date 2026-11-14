"""
Tests for the specifications of steps and the context they share; how a stepped job runs them is in
``test_jobs.py``.
"""

import pytest

from xtl.jobs.config import BatchJobConfig, JobConfig
from xtl.jobs.jobs import BatchJob, Job
from xtl.jobs.steps import JobContext, StepSpec
from xtl_core.options import Option


class ValueConfig(JobConfig):
    value: int = Option(default=0, desc='Returned by the job')
    tags: list[str] = Option(default_factory=list, desc='Returned by the job')


class ReturnsValue(Job[ValueConfig]):
    async def _execute(self):
        return self.config.value


class TestStepSpec:
    def test_defaults(self):
        spec = StepSpec(name='a', job_cls=ReturnsValue, config_cls=ValueConfig)
        ctx = JobContext(parent=ReturnsValue(job_id='p'))
        assert spec.condition(ctx) is True
        assert spec.dynamic_defaults(ctx) == {}
        assert spec.defaults == {}
        assert spec.post_processor is None
        assert spec.desc is None

    def test_frozen(self):
        spec = StepSpec(name='a', job_cls=ReturnsValue, config_cls=ValueConfig)
        with pytest.raises(Exception):
            spec.name = 'b'

    def test_job_cls_must_be_a_job(self):
        with pytest.raises(Exception):
            StepSpec(name='a', job_cls=int, config_cls=JobConfig)

    def test_config_cls_must_fit_the_job(self):
        with pytest.raises(Exception, match='config_cls must be a subclass of BatchJobConfig'):
            StepSpec(name='a', job_cls=BatchJob, config_cls=JobConfig)
        # A subclass of the job's configuration is fine
        StepSpec(name='a', job_cls=Job, config_cls=ValueConfig)
        StepSpec(name='b', job_cls=BatchJob, config_cls=BatchJobConfig)

    def test_repr(self):
        spec = StepSpec(name='a', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 1})
        assert repr(spec).startswith("StepSpec<ReturnsValue[ValueConfig]>(name='a', defaults={'value': 1}")
        rich = list(spec.__rich_repr__())
        assert rich[0] == '<ReturnsValue[ValueConfig]>'
        assert ('name', 'a') in rich


class TestJobContext:
    def test_defaults_not_shared(self):
        parent = ReturnsValue(job_id='p')
        first, second = JobContext(parent=parent), JobContext(parent=parent)
        first.results['a'] = 1
        first.data['b'] = 2
        assert second.results == {} and second.data == {}
        assert first.step is None
