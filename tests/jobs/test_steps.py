"""
Tests for the specifications of steps, and how a stepped job runs them.
"""

import pytest

from tests.jobs.helpers import Failing
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.steps import JobContext, StepSpec
from xtl_core.options import Option


class ValueConfig(JobConfig):
    value: int = Option(default=0, desc='Returned by the job')
    tags: list[str] = Option(default_factory=list, desc='Returned by the job')


class ReturnsValue(Job[ValueConfig]):
    async def _execute(self):
        return self.config.value


class ReturnsTags(Job[ValueConfig]):
    async def _execute(self):
        self.config.tags.append('touched')  # must not reach the spec's defaults
        return self.config.tags


def run(job_cls, steps=None, job_directory=None):
    job = job_cls(job_id=job_cls.__name__)
    job.configure(SteppedJobConfig(steps=steps or {}, job_directory=job_directory))
    return job.run()


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


class Configured(SteppedJob[SteppedJobConfig]):
    """Defaults, dynamic defaults from the first step's result, overrides, a condition and a post-processor."""

    _steps = (
        StepSpec(name='first', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 1}),
        StepSpec(
            name='second',
            job_cls=ReturnsValue,
            config_cls=ValueConfig,
            defaults={'value': 2},
            dynamic_defaults=lambda ctx: {'value': ctx.results['first'].data * 10},
            post_processor=lambda results, ctx: {'second_doubled': results.data * 2},
        ),
        StepSpec(
            name='skipped',
            job_cls=Failing,
            config_cls=JobConfig,
            condition=lambda ctx: ctx.data.get('second_doubled') == 0,
        ),
    )


class TestSteppedJob:
    @pytest.mark.asyncio
    async def test_defaults_dynamic_defaults_and_overrides(self):
        result = await run(Configured)
        assert result.success
        assert {name: r.data for name, r in result.steps.items()} == {'first': 1, 'second': 10}
        assert result.data == {'second_doubled': 20}

        # Overrides win over both kinds of defaults
        result = await run(Configured, steps={'first': {'value': 3}, 'second': {'value': 7}})
        assert {name: r.data for name, r in result.steps.items()} == {'first': 3, 'second': 7}

    @pytest.mark.asyncio
    async def test_skipped_step_has_no_result(self):
        result = await run(Configured)
        assert 'skipped' not in result.steps

    @pytest.mark.asyncio
    async def test_defaults_are_copied(self):
        class Twice(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='a', job_cls=ReturnsTags, config_cls=ValueConfig, defaults={'tags': ['x']}),
                StepSpec(name='b', job_cls=ReturnsTags, config_cls=ValueConfig, defaults={'tags': ['x']}),
            )

        result = await run(Twice)
        assert [r.data for r in result.steps.values()] == [['x', 'touched'], ['x', 'touched']]
        assert Twice._steps[0].defaults == {'tags': ['x']}

    @pytest.mark.asyncio
    async def test_failing_step_stops_the_job(self):
        class Stops(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='ok', job_cls=ReturnsValue, config_cls=ValueConfig),
                StepSpec(name='fails', job_cls=Failing, config_cls=JobConfig),
                StepSpec(name='never', job_cls=ReturnsValue, config_cls=ValueConfig),
            )

        result = await run(Stops)
        assert not result.success
        assert list(result.steps) == ['ok', 'fails']
        assert isinstance(result.error, ValueError) and result.error is result.steps['fails'].error

    @pytest.mark.asyncio
    async def test_step_directories(self, tmp_path):
        class Keeps(SteppedJob[SteppedJobConfig]):
            _steps = (StepSpec(name='only', job_cls=ReturnsValue, config_cls=ValueConfig),)

        job = Keeps(job_id='k')
        job.configure(SteppedJobConfig(job_directory=tmp_path))
        ctx_dirs = []
        original = job._create_child

        def spy(job_cls, config, suffix, **kwargs):
            ctx_dirs.append(config.job_directory)
            return original(job_cls=job_cls, config=config, suffix=suffix, **kwargs)

        job._create_child = spy
        await job.run()
        assert ctx_dirs == [tmp_path / '1_only']

    def test_duplicate_step_names(self):
        with pytest.raises(ValueError, match='Duplicate step names in Duplicate._steps: same'):

            class Duplicate(SteppedJob[SteppedJobConfig]):
                _steps = (
                    StepSpec(name='same', job_cls=ReturnsValue, config_cls=ValueConfig),
                    StepSpec(name='same', job_cls=ReturnsValue, config_cls=ValueConfig),
                )


def _raises(*args):
    raise RuntimeError('hook failed')


class TestEarlierStepsKept:
    """A step that fails before or after its job runs keeps the results of the steps before it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('hook', ['condition', 'dynamic_defaults', 'post_processor'])
    async def test_hook_raises(self, hook):
        class Hooked(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='done', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 5}),
                StepSpec(name='hooked', job_cls=ReturnsValue, config_cls=ValueConfig, **{hook: _raises}),
            )

        result = await run(Hooked)
        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert result.steps['done'].data == 5

    @pytest.mark.asyncio
    async def test_bad_option_value(self):
        class Later(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='done', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 5}),
                StepSpec(name='bad', job_cls=ReturnsValue, config_cls=ValueConfig),
            )

        # Option names are checked when the job is configured, values when the step's configuration is built
        result = await run(Later, steps={'bad': {'value': 'not a number'}})
        assert not result.success
        assert result.steps['done'].data == 5
