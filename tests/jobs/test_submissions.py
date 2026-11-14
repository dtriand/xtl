import logging
import pickle

import pytest

from tests.jobs.helpers import Talker
from xtl import settings
from xtl.config.settings import XTLSettings
from xtl.jobs.config import JobConfig
from xtl.jobs.resources import Resources
from xtl.jobs.submissions import JobSubmission


@pytest.fixture
def restore_settings():
    """Put back the settings that a test applies a snapshot over."""
    before = settings.to_dict()
    yield
    settings.update_from(XTLSettings.from_dict(before))


def talker(job_id: str = 'job', level: int = logging.INFO) -> Talker:
    logger = logging.getLogger(f'Pool#1.{job_id}')
    logger.setLevel(level)
    return Talker.with_config(JobConfig(), job_id=job_id, logger=logger)


class TestJobSubmission:
    def test_from_job(self):
        submission = JobSubmission.from_job(talker('t1', level=logging.DEBUG))
        assert submission.data.job_id == 't1'
        assert submission.log_level == logging.DEBUG
        assert submission.logger_name == 'Pool#1.t1'
        assert submission.job_logger_name == 'Pool#1.t1'
        assert (submission.ipc, submission.resources, submission.xtl_settings) == (None, None, None)

    def test_submission_ids(self, monkeypatch):
        monkeypatch.setattr(settings.jobs, 'job_digits', 6)
        first, second = JobSubmission.from_job(talker()), JobSubmission.from_job(talker())
        assert len(first.submission_id) == 6
        assert first.submission_id != second.submission_id

    def test_job_logger_name_defaults_to_the_job_id(self):
        submission = JobSubmission.from_job(talker('t2'))
        submission.logger_name = None
        assert submission.job_logger_name == 't2'

    def test_round_trip(self):
        submission = JobSubmission.from_job(talker('t3'))
        submission.resources = Resources(jobs=2)
        submission.xtl_settings = settings.to_dict()
        for restored in (
            JobSubmission.from_dict(submission.to_dict()),
            JobSubmission.from_dict(pickle.loads(pickle.dumps(submission.to_dict()))),  # as to a worker
        ):
            assert restored.submission_id == submission.submission_id
            assert restored.data.job_id == 't3'
            assert restored.resources == submission.resources
            assert restored.xtl_settings == submission.xtl_settings
            assert (restored.log_level, restored.logger_name) == (submission.log_level, submission.logger_name)

    def test_to_job(self):
        logger = logging.getLogger('Pool#2.t4')
        job = JobSubmission.from_job(talker('t4')).to_job(logger=logger)
        assert isinstance(job, Talker)
        assert job.job_id == 't4'
        assert job.logger is logger
        assert isinstance(job.config, JobConfig)

    def test_apply_settings(self, restore_settings):
        submission = JobSubmission.from_job(talker())
        submission.xtl_settings = settings.to_dict()
        submission.xtl_settings['jobs']['keep_temp'] = not settings.jobs.keep_temp
        jobs = settings.jobs
        submission.apply_settings()
        # In place: whoever holds the settings sees the parent's values
        assert settings.jobs is jobs
        assert settings.jobs.keep_temp == submission.xtl_settings['jobs']['keep_temp']

    def test_apply_settings_without_a_snapshot(self, restore_settings):
        before = settings.to_dict()
        JobSubmission.from_job(talker()).apply_settings()
        assert settings.to_dict() == before
