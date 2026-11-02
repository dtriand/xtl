import abc
from pathlib import Path

from xtl.common.options import Option, Options
from xtl.jobs.jobs import Job, JobConfig


class RefinementJobConfig(JobConfig):
    structure_file: str | Path = Option(desc='Structure file to refine')
    reflections_file: str | Path = Option(desc='Reflections file for refinement')

    job_dir: str | Path = Option(default='.', desc='Directory to output job files')

    executable: str = Option(desc='Executable name')


class RefinementJob(Job[RefinementJobConfig], abc.ABC): ...


class RefmacJobConfig(RefinementJobConfig):
    executable: str = Option(default='refmac', desc='Executable name')


class RefmacJob(Job[RefmacJobConfig]):
    def _execute(self): ...


class PhenixRefineJobConfig(RefinementJobConfig):
    executable: str = Option(default='phenix.refine', desc='Executable name')


class PhenixRefineJob(Job[PhenixRefineJobConfig]):
    def _execute(self): ...