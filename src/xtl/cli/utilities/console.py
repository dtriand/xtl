import logging
from typing import TYPE_CHECKING

import rich.console
import rich.theme

from xtl import settings
from xtl.logging.config import LoggerConfig

if TYPE_CHECKING:
    from xtl.jobs.pools import JobPool
    from xtl.cli.utilities.common import AutomateOptions


class ConsoleIO(rich.console.Console):

    STYLES: dict[str, str] = {
        'xtl.jobs.job.job_id': 'blue',
        'xtl.jobs.pool.name': 'magenta',
    }

    class logger_config:

        JOB: LoggerConfig
        POOL: LoggerConfig

    class loggers:

        POOL: logging.Logger


    def __init__(self, *args, verbose: int = 0, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.debug = debug
        self.push_theme(theme=rich.theme.Theme(styles=self.STYLES))

    def _setup_job_logging(self):
        if hasattr(self.loggers, 'POOL'):
            return

        from xtl.cli.utilities.logging import JobLoggerConfig, JobPoolLoggerConfig

        # Configuration
        self.logger_config.JOB = JobLoggerConfig(console=self)
        self.logger_config.JOB.level = logging.DEBUG if self.debug else logging.INFO

        self.loggers.POOL = logging.getLogger('xtl.jobs.pools')
        self.logger_config.POOL = JobPoolLoggerConfig(console=self)
        self.logger_config.POOL.level = logging.DEBUG if self.debug else logging.INFO
        self.logger_config.POOL.handlers[0].format.format = '[JobPool] %(message)s'

    def get_pool(self, max_jobs: int = 1) -> 'JobPool':
        """
        Returns a JobPool instance with a configured logger attached to the console.
        """
        from xtl.jobs.pools import JobPool

        self._setup_job_logging()
        return JobPool(max_jobs=max_jobs, logger_config=self.logger_config.JOB)

    def report_automate(self, options: 'AutomateOptions') -> None:
        if self.verbose:
            from xtl.common.compatibility import OS_POSIX

            self.print(f'Using compute site: [dim]{options.compute_site}[/]')
            if options.permissions.update and OS_POSIX:
                self.print(f'Updating permissions to: [dim]'
                           f'{options.permissions.files.octal[2:]} (files) and '
                           f'{options.permissions.directories.octal[2:]} (directories)',
                           highlight=False)
            if options.compute_site == 'modules' and options.modules:
                self.print(f'Using modules: '
                           f'[dim]{", ".join(options.modules)}[/]',
                           highlight=False)
