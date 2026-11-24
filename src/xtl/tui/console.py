from typing import Iterable, Optional, TYPE_CHECKING

import rich.console

if TYPE_CHECKING:
    from xtl.jobs.pools import PoolType
    from xtl.cli.utilities.common import JobOptions
    from xtl.tui.live_pool import LivePool


class ConsoleIO(rich.console.Console):
    def __init__(self, *args, verbose: int = 0, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.debug = debug

    def confirm(self, question: str, **kwargs) -> bool:
        import rich.prompt

        return rich.prompt.Confirm.ask(question, console=self, **kwargs)

    def apply_job_options(self, options: 'JobOptions') -> None:
        """
        Update job related options globally and report the settings used.

        :param options: Job options overrides.
        """
        from xtl import settings
        from xtl.cli.utilities.decorators import CLI_DEPENDENCIES

        # Modify runtime settings
        settings.jobs.compute_site = options.compute_site
        settings.jobs.permissions.update = options.permissions.update
        settings.jobs.permissions.files = options.permissions.files
        settings.jobs.permissions.directories = options.permissions.directories
        settings.jobs.keep_temp = options.keep_temp

        # Inject modules to settings: those of a dependency replace its modules (and its path, which can't be set
        #  together with them), the rest the common modules
        if options.has_modules:
            common = []
            for dep_name, modules in options.modules.items():
                dependency = settings.dependencies.get(dep_name)
                if dependency is not None:
                    dependency.path = None
                    dependency.modules = modules
                else:
                    common.extend(modules)
            if common:
                settings.dependencies.modules = common

        # Report job settings
        if self.verbose >= 1:
            from xtl_core.system.compatibility import OS_POSIX

            self.print(f'Using compute site: [dim]{options.compute_site}[/]', highlight=False)
            if options.permissions.update and OS_POSIX:
                self.print(
                    f'Updating permissions to: [dim]'
                    f'{options.permissions.files.octal[2:]} (files), '
                    f'{options.permissions.directories.octal[2:]} (directories)',
                    highlight=False,
                )
            # The modules of the command's dependencies (see `requires`)
            dependencies = {name: settings.dependencies.get(name).modules for name in sorted(CLI_DEPENDENCIES.get())}
            if options.compute_site == 'modules' and (settings.dependencies.modules or any(dependencies.values())):
                self.print('Using modules: ', highlight=False)
                if settings.dependencies.modules:
                    self.print(f'- all jobs: [dim]{", ".join(settings.dependencies.modules)}[/dim]', highlight=False)
                for dep, modules in dependencies.items():
                    if modules:
                        self.print(f'- {dep}: [dim]{", ".join(modules)}[/dim]', highlight=False)

    def check_dependencies(self, dependencies: Optional[Iterable[str]] = None) -> None:
        """
        Check that the command's dependencies can be found, once, before it starts any jobs: each with the compute
        site and shell that a job needing it would use (see :func:`~xtl.jobs.dependencies.select_site_and_shell`).
        The outcome is remembered in the settings, so pools and jobs don't check again. Call it after
        :meth:`apply_job_options`, which may change the compute site and modules. Nothing is checked if
        ``settings.dependencies.check`` is off.

        If a dependency is missing, prints what is missing (and what the check printed, with ``-v``) and exits with
        code 1.

        :param dependencies: Names of dependencies (default: those the command declares with
            :func:`~xtl.cli.utilities.decorators.requires`).
        :raises typer.Exit: If a dependency is missing.
        """
        from rich.markup import escape

        from xtl import settings
        from xtl.cli.utilities.decorators import CLI_DEPENDENCIES
        from xtl.jobs.dependencies import select_site_and_shell, unknown_dependencies
        from xtl.jobs.exceptions import DependencyError

        names = tuple(dependencies) if dependencies is not None else CLI_DEPENDENCIES.get()
        if not names or not settings.dependencies.check:
            return

        # One line per problem, e.g. `module atsas/9.9 not found (required for atsas)`, and what each check printed
        problems = [
            f'unknown dependency {name} (not in the [dependencies] of xtl.toml)' for name in unknown_dependencies(names)
        ]
        details: list[tuple[str, str]] = []
        for name in (name for name in names if name in settings.dependencies):
            try:
                site, shell = select_site_and_shell([name])
            except (DependencyError, ValueError) as error:
                problems.append(f'{name} could not be checked: {error}')
                continue
            if self.verbose >= 1:
                self.print(
                    f'Checking dependency: [dim]{name} ({type(site).__name__}, {shell.name})[/]', highlight=False
                )
            for check in site.check_dependencies([name], shell=shell):
                problems.extend(check.problems())
                if not check.found and check.details:
                    details.append((f'{name} ({type(site).__name__}, {shell.name})', check.details))

        if not problems:
            return
        self.print('Missing dependencies:', style='bold red')
        for problem in problems:
            self.print(f'- {escape(problem)}', style='red', highlight=False)
        if self.verbose >= 1:
            for label, text in details:
                self.print(f'What the check of {label} printed:', style='bold', highlight=False)
                self.print(escape(text), style='dim', highlight=False)
        elif details:
            self.print('Run with -v to see what the checks printed.', highlight=False)
        self.print(
            'Check [b]dependencies[/] in xtl.toml (xtl.config where, xtl.config deps), or use --compute-site/--modules.'
        )

        import typer

        raise typer.Exit(code=1)

    def get_pool(self, pool_type: Optional['PoolType'] = None, max_jobs: int = 1) -> 'LivePool':
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

    def print_traceback(self, exception: Exception, *, message: str | None = None) -> None:
        if self.debug:
            # Rich traceback
            try:
                raise exception
            except Exception:
                self.print_exception(show_locals=True)
        elif self.verbose:
            # Standard traceback
            import traceback

            for line in traceback.format_exception(type(exception), exception, exception.__traceback__):
                self.print(line, style='red')
        else:
            # Only exception message
            self.print(f'{exception}', style='red')
        if message:
            self.print(f'{message}', style='red')
