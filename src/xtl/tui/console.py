from typing import Literal, TYPE_CHECKING, Union

import rich.console
import rich.prompt

if TYPE_CHECKING:
    from xtl.jobs import JobPool
    from xtl.cli.utilities.common import JobOptions, REQUIRED_DEPENDENCIES
    from xtl.tui.live_pool import LivePool


class ConsoleIO(rich.console.Console):

    def __init__(self, *args, verbose: int = 0, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.debug = debug

    def confirm(self, question: str, **kwargs) -> bool:
        return rich.prompt.Confirm.ask(question, console=self, **kwargs)

    def apply_job_options(self, options: 'JobOptions') -> None:
        """
        Update job related options globally and report the settings used.

        :param options: Job options overrides.
        """
        from xtl import settings
        from xtl.cli.utilities.common import REQUIRED_DEPENDENCIES

        # Modify runtime settings
        settings.jobs.compute_site = options.compute_site
        settings.jobs.permissions.update = options.permissions.update
        settings.jobs.permissions.files = options.permissions.files
        settings.jobs.permissions.directories = options.permissions.directories
        settings.jobs.keep_temp = options.keep_temp

        # Inject modules to settings
        if options.has_modules:
            for dep_name, modules in options.modules.items():
                if dep_name in settings.dependencies.to_dict().keys():
                    REQUIRED_DEPENDENCIES[dep_name] = modules
                    dep = getattr(settings.dependencies, dep_name)
                    dep.modules = modules
                else:
                    REQUIRED_DEPENDENCIES['extra'].extend(modules)

        # Report job settings
        if self.verbose >= 1:
            from xtl.common.compatibility import OS_POSIX

            self.print(f'Using compute site: [dim]{options.compute_site}[/]', highlight=False)
            if options.permissions.update and OS_POSIX:
                self.print(f'Updating permissions to: [dim]'
                           f'{options.permissions.files.octal[2:]} (files), '
                           f'{options.permissions.directories.octal[2:]} (directories)',
                           highlight=False)
            if options.compute_site == 'modules' and any(REQUIRED_DEPENDENCIES.values()):
                self.print(f'Using modules: ', highlight=False)
                for dep, modules in sorted(REQUIRED_DEPENDENCIES.items()):
                    if modules:
                        self.print(f'- {dep}: [dim]{", ".join(modules)}[/dim]', highlight=False)


    def get_pool(self, pool_type: Union['JobPool', Literal['simple', 'async', 'threads', 'processes']] = None,
                 max_jobs: int = 1) -> 'LivePool':
        """
        Get a pool for job execution which redirects job logs and status updates to a transient Live view on the
        console.

        :param pool_type: Type of pool to use.
        :param max_jobs: Max number of jobs to run concurrently in the pool.
        """
        from xtl.tui.live_pool import LivePool

        return LivePool(
            pool_type=pool_type,
            max_jobs=max_jobs,
            console=self,
        )

