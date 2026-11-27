import dataclasses

import pytest

from xtl.jobs.results import BatchResults, JobResults, JobStatus, SteppedJobResults


class TestJobStatus:
    @pytest.mark.parametrize(
        'status, finished',
        [
            (JobStatus.SUBMITTED, False),
            (JobStatus.QUEUED, False),
            (JobStatus.RUNNING, False),
            (JobStatus.DONE, True),
            (JobStatus.FAILED, True),
            (JobStatus.CANCELLED, True),
        ],
    )
    def test_is_finished(self, status, finished):
        assert status.is_finished is finished

    def test_values_are_strings(self):
        assert JobStatus('running') is JobStatus.RUNNING
        assert JobStatus.DONE == 'done'
        assert f'{JobStatus.FAILED}' == 'failed'


class TestJobResults:
    def test_success_without_error(self):
        result = JobResults(job_id='a', data={'x': 1})
        assert result.success
        assert result.data == {'x': 1}

    def test_failure_with_error(self):
        result = JobResults(job_id='a', error=ValueError('boom'))
        assert not result.success
        assert result.data is None

    def test_falsy_data_is_still_a_success(self):
        assert JobResults(job_id='a', data=0).success


class TestBatchResults:
    def test_frozen(self):
        result = BatchResults(stdout='out', stderr='', return_code=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.return_code = 1


class TestSteppedJobResults:
    def test_steps(self):
        steps = {'one': JobResults(job_id='a.1'), 'two': JobResults(job_id='a.2', error='failed')}
        result = SteppedJobResults(job_id='a', steps=steps)
        assert isinstance(result, JobResults)
        assert result.success  # the stepped job's own error decides
        assert not result.steps['two'].success

    def test_steps_not_shared(self):
        first, second = SteppedJobResults(job_id='a'), SteppedJobResults(job_id='b')
        first.steps['one'] = JobResults(job_id='a.1')
        assert second.steps == {}
