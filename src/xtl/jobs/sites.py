from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import contextlib
import logging
from enum import Enum
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from typing import Optional, Iterable, Sequence, TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from xtl.jobs.config import BatchJobConfig
    from xtl.config.settings import DependencySettings
from xtl_core.system.compatibility import OS_POSIX, OS_WINDOWS, PY310_OR_LESS
from xtl.shared.environment import XTL_COMPUTE_SITE
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus
from xtl.jobs.dependencies import DependencyCheck, dependency_name
from xtl.jobs.policies import CommandPolicy, CommandPolicyType
from xtl.jobs.shells import CHECK_MARKER, Shell, ShellType
from xtl.jobs.exceptions import SubprocessError


__all__ = ['ComputeSite', 'ComputeSiteType', 'BaseComputeSite', 'LocalSite', 'ModulesSite']


logger = logging.getLogger(__name__)


if PY310_OR_LESS:

    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class ComputeSite(StrEnum):
    # General purpose compute sites
    LOCAL = 'local'
    MODULES = 'modules'

    # Specialized compute sites
    if XTL_COMPUTE_SITE.value == 'BIOTIX_HPC':
        BIOTIX = 'biotix'

    def get(self, **kwargs) -> ComputeSiteType:
        """
        An instance of the compute site.

        :param kwargs: Passed to the site, e.g. ``policy`` (see :class:`BaseComputeSite`).
        """
        mapping: dict[str, type] = {
            ComputeSite.LOCAL: LocalSite,
            ComputeSite.MODULES: ModulesSite,
        }
        if XTL_COMPUTE_SITE.value == 'BIOTIX_HPC':
            mapping[ComputeSite.BIOTIX] = BiotixHPC
        return mapping[self](**kwargs)

    def __eq__(self, other):
        if isinstance(other, BaseComputeSite):
            return self.get() == other
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class BaseComputeSite(ABC):
    _default_shell: Shell = None
    _supported_shells: frozenset[Shell] = frozenset()
    _policy: Optional[CommandPolicyType] = None

    def __init__(
        self,
        default_shell: Optional[Shell | ShellType] = None,
        shells: Optional[Iterable[Shell | ShellType]] = None,
        policy: Optional[CommandPolicy | CommandPolicyType] = None,
    ):
        """
        :param default_shell: The shell to prefer, instead of the site's own default
        :param shells: The shells that the site supports, instead of its own
        :param policy: The command policy, instead of the site's own
        :raises ValueError: If the default shell isn't one of the supported shells
        """
        if shells is not None:
            self._supported_shells = frozenset(self._as_shell(shell) for shell in shells)
        if default_shell is not None:
            self._default_shell = self._as_shell(default_shell)
            if not self.is_valid_shell(self._default_shell):
                raise ValueError(f'Shell {self._default_shell!r} is not one of the shells that the site supports')
        if policy is not None:
            self._policy = policy.get() if isinstance(policy, CommandPolicy) else policy

    @staticmethod
    def _as_shell(shell: Shell | ShellType) -> Shell:
        # Shells as enum members, which hash alike in sets
        return Shell(shell.name) if not isinstance(shell, Shell) else shell

    @property
    def default_shell(self) -> Optional[Shell]:
        """
        Returns the default shell for this compute site. This shell is preferred over
        other compatible shells when executing commands. If no default shell is defined,
        None is returned.

        :return: The default Shell instance or None.
        """
        return self._default_shell

    @property
    def supported_shells(self) -> frozenset[Shell]:
        return self._supported_shells

    @property
    def policy(self) -> Optional[CommandPolicyType]:
        return self._policy

    def __eq__(self, other):
        if isinstance(other, ComputeSite):
            return self == other.get()
        elif isinstance(other, BaseComputeSite):
            return (
                type(self) is type(other)
                and self.default_shell == other.default_shell
                and self.supported_shells == other.supported_shells
                and self.policy == other.policy
            )
        return NotImplemented

    def is_valid_shell(self, shell: Shell | ShellType) -> bool:
        if not self.supported_shells:
            return True
        return self._as_shell(shell) in {self._as_shell(supported) for supported in self.supported_shells}

    @abstractmethod
    def prepare_preamble(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> str: ...

    @abstractmethod
    def prepare_command(self, command: str) -> str: ...

    @abstractmethod
    def prepare_content(self, content: str) -> str: ...

    @abstractmethod
    def prepare_postamble(self) -> str: ...

    @staticmethod
    def resolve_dependencies(
        dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None,
    ) -> list[DependencySettings]:
        """
        Resolves dependencies from settings based on the provided dependency names or
        DependencySettings objects.

        :param dependencies: An iterable of dependency names (str) or DependencySettings
            instances, or None.
        :return: The resolved DependencySettings instances, without duplicates.
        :raises ValueError: If a dependency name is not found in settings and strict
            resolution is enabled.
        """
        # Imports to avoid circular dependencies
        from xtl import settings

        return settings.dependencies.resolve(dependencies)

    @abstractmethod
    def check_dependencies(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> list[DependencyCheck]: ...

    @staticmethod
    def execute_command(command: Sequence[str], timeout: int = 20) -> subprocess.CompletedProcess | None:
        """
        Execute a command locally and return the CompletedProcess object.

        :param command: The command to execute as a list of strings.
        :param timeout: The timeout in seconds for the command execution.
        :return: The CompletedProcess object if the command executed successfully,
            None if the command timed out or couldn't be started (e.g. the program
            doesn't exist).
        """
        try:
            # NB: Output that isn't text in the expected encoding (e.g. a program's banner) is replaced, not an error
            result = subprocess.run(
                command, capture_output=True, text=True, errors='replace', shell=False, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            logger.error('Command `%s` timed out', ' '.join(command))
            return None
        except OSError as e:
            logger.error('Command `%s` could not be started: %s', ' '.join(command), e)
            return None
        return result

    @abstractmethod
    async def execute_batch(self, batch: BatchFile, **kwargs): ...

    @staticmethod
    def _prepare_shebang(shell: ShellType) -> str:
        if shell.shebang:
            return f'{shell.shebang}{shell.new_line_char}'
        return ''


async def _uninterrupted(coro) -> None:
    """
    Run a coroutine to its end, even if the task that awaits it is cancelled meanwhile; such a cancellation is raised
    once the coroutine is done.
    """
    task = asyncio.ensure_future(coro)
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


class LocalSite(BaseComputeSite):
    """
    A compute site that represents the local machine. The executables of dependencies are found on ``PATH``, after
    the directories of ``settings.dependencies.path`` and of each dependency's ``path`` are put first on it.
    """

    _CHECK_TIMEOUT: int = 60
    """Seconds that the check of a dependency may take, e.g. to load its modules."""

    def prepare_setup(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> str:
        """
        The lines that set up dependencies in a shell, each ending with the shell's line ending: on this site, the
        common ``path`` and then each dependency's are put first on ``PATH`` (so a dependency's directory comes first).

        :param dependencies: Dependency names or settings, or None.
        :param shell: The shell to write the lines for.
        :raises ValueError: If a name or a directory can't be written safely in the shell.
        """
        lines = self._setup_lines(self.resolve_dependencies(dependencies), shell)
        return ''.join(line + shell.new_line_char for line in lines)

    def _setup_lines(self, dependencies: list[DependencySettings], shell: ShellType, report: bool = False) -> list[str]:
        """
        The lines of :meth:`prepare_setup`; with ``report``, as a check runs them, reporting what didn't work.
        """
        from xtl import settings

        directories = [settings.dependencies.path, *(dep.path for dep in dependencies)]
        return [shell.get_path_command(directory) for directory in directories if directory is not None]

    @staticmethod
    def _guard_lines(dependencies: list[DependencySettings], shell: ShellType) -> list[str]:
        """
        One line per executable of the dependencies, which ends the batch file if the executable isn't found.
        """
        executables = sorted({exe for dep in dependencies for exe in dep.provides})
        return [shell.get_guard_command(exe) for exe in executables]

    def _prepare_directives(self, shell: ShellType, **kwargs) -> str:
        """
        Lines that follow the shebang, before the setup, e.g. a scheduler's directives; none on this site.
        """
        return ''

    def prepare_preamble(
        self,
        dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None,
        shell: ShellType,
        **kwargs,
    ) -> str:
        """
        The preamble of a batch file: the shebang, the site's directives (if any), the setup of the dependencies
        (see :meth:`prepare_setup`), and a guard line for each executable they provide, which ends the batch file with
        exit code 127 if the executable isn't found, before any command runs. Then passed through the policy, if any.

        :param dependencies: Dependency names or settings, or None.
        :param shell: The shell to write the preamble for.
        :param kwargs: Passed to the site's directives, e.g. ``config`` on a Slurm site.
        :raises ValueError: If a name or a directory can't be written safely in the shell.
        """
        resolved = self.resolve_dependencies(dependencies)
        lines = [*self._setup_lines(resolved, shell), *self._guard_lines(resolved, shell)]
        preamble = self._prepare_shebang(shell) + self._prepare_directives(shell, **kwargs)
        preamble += ''.join(line + shell.new_line_char for line in lines)
        return self.policy.intercept_preamble(preamble) if self.policy else preamble

    def prepare_command(self, command: str) -> str:
        return self.policy.intercept_command(command) if self.policy else command

    def prepare_content(self, content: str) -> str:
        return self.policy.intercept_content(content) if self.policy else content

    def prepare_postamble(self) -> str:
        return self.policy.intercept_postamble('') if self.policy else ''

    def check_dependencies(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> list[DependencyCheck]:
        """
        Check that dependencies can be found in a shell: for each dependency, a short script in the shell sets up the
        common settings and the dependency's own (see :meth:`prepare_setup`), then looks up every executable that the
        dependency provides. A dependency is found only if all of them are; a module that doesn't load is reported
        too. Each dependency is checked on its own, and the outcome is remembered in its settings, so that it isn't
        checked again on this site and in this shell until its settings change.

        :param dependencies: Dependency names or settings, or None.
        :param shell: The shell to check in.
        :return: One check per dependency, in the given order.
        """
        checks = []
        for dep in self.resolve_dependencies(dependencies):
            check = dep.get_check(self, shell)
            if check is None:
                check = self._check_dependency(dep, shell)
                dep.remember_check(self, shell, check)
            checks.append(check)
        return checks

    def _check_dependency(self, dependency: DependencySettings, shell: ShellType) -> DependencyCheck:
        name = dependency_name(dependency)
        executables = sorted(dependency.provides)
        try:
            lines = [
                *self._setup_lines([dependency], shell, report=True),
                *(shell.get_found_report_command(exe) for exe in executables),
            ]
        except ValueError as e:
            logger.error('Cannot check dependency `%(dependency)s`: %(error)s', {'dependency': name, 'error': e})
            return DependencyCheck(dependency=name, found=False, error=str(e))

        script = self._prepare_shebang(shell) + ''.join(line + shell.new_line_char for line in lines)
        result = self._run_script(script, shell)
        if result is None:
            return DependencyCheck(
                dependency=name,
                found=False,
                error=f'the check in {shell.name} did not finish within {self._CHECK_TIMEOUT} s, or could not start',
            )

        # What the lines reported, by kind (command, module, executable) and name; the rest is what the setup printed
        reported, printed = {}, []
        for line in result.stdout.splitlines():
            words = line.split()
            if len(words) == 4 and words[0] == CHECK_MARKER:
                reported[(words[1], words[2])] = words[3]
            elif line.strip():
                printed.append(line.rstrip())
        missing_command = next(
            (command for (kind, command), code in reported.items() if kind == 'command' and code != '0'), ''
        )
        missing_modules = tuple(m for (kind, m), code in reported.items() if kind == 'module' and code != '0')
        missing_executables = tuple(exe for exe in executables if reported.get(('executable', exe)) != '0')
        found = not (missing_command or missing_modules or missing_executables)

        details = '\n'.join([*printed, *(line.rstrip() for line in result.stderr.splitlines() if line.strip())])
        check = DependencyCheck(
            dependency=name,
            found=found,
            missing_executables=missing_executables,
            missing_modules=missing_modules,
            missing_command=missing_command,
            details=details,
        )
        if found:
            logger.debug('Found dependency `%(dependency)s`', {'dependency': name})
        else:
            logger.debug(
                'Dependency `%(dependency)s`: %(problems)s', {'dependency': name, 'problems': check.problems()}
            )
        return check

    def _run_script(self, script: str, shell: ShellType) -> subprocess.CompletedProcess | None:
        """
        Run a script in a shell, as a batch file would run, from a temporary file.
        """
        fd, file = tempfile.mkstemp(prefix='xtl-check-', suffix=shell.batch_extension)
        try:
            with os.fdopen(fd, 'w', encoding=shell.file_encoding, newline='') as f:
                f.write(script)
            command = shell.get_execute_batch_command(file, as_list=True)
            return self.execute_command(command=command, timeout=self._CHECK_TIMEOUT)
        finally:
            Path(file).unlink(missing_ok=True)

    async def execute_batch(self, batch: BatchFile, stdout: Path = None, stderr: Path = None, **kwargs):
        # Get command to execute the batch file
        args = kwargs.pop('batch_args', [])
        cmd = batch.shell.get_execute_batch_command(batch.file, arguments=args, as_list=True)

        if PY310_OR_LESS:
            # TODO: Update exception handling to asyncio.TaskGroup when we drop support
            #  for Python 3.10
            pass

        # Initialize stream logging tasks
        stdout_task, stderr_task = None, None
        try:
            # Start batch execution; in a session (POSIX) or process group (Windows) of its own, so that the programs
            #  it starts can be stopped together (see `_terminate_batch`), and that Ctrl+C on the terminal or console
            #  reaches only the application, which cancels the job and stops them. Without input: a batch file isn't interactive, and the console's
            #  input belongs to the application (e.g. cmd's "Terminate batch job (Y/N)?" after Ctrl+C would wait for
            #  the keyboard, competing with the application's own prompts)
            batch._process = await asyncio.create_subprocess_exec(
                *cmd,
                shell=False,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=OS_POSIX,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if OS_WINDOWS else 0,
            )
            batch._status = BatchFileStatus.RUNNING

            # Start logging of streams
            stdout_task = asyncio.create_task(
                self._log_stream_to_file(batch.process.stdout, stdout), name='stdout_logger'
            )
            stderr_task = asyncio.create_task(
                self._log_stream_to_file(batch.process.stderr, stderr), name='stderr_logger'
            )

            # Wait for all tasks to complete. This is where the main thread waits.
            await asyncio.gather(batch.process.wait(), stdout_task, stderr_task)

            # Check if the batch was cancelled by .cancel_batch()
            if batch.status != BatchFileStatus.CANCELLED:
                failed = batch.process.returncode != 0
                batch._status = BatchFileStatus.FAILED if failed else BatchFileStatus.COMPLETED
        except asyncio.CancelledError as e:
            # Raise cancellation up the stack after marking the batch as cancelled; said before the clean-up, which
            #  takes a while
            batch.logger.warning('Stopping the batch file and the programs it started')
            batch._status = BatchFileStatus.CANCELLED
            raise e
        except Exception as e:
            # Log any other exceptions during execution
            batch.logger.error('Error executing batch file %(file)s: %(exc)s', {'file': batch.file, 'exc': str(e)})
            batch._status = BatchFileStatus.FAILED
            raise SubprocessError(f'Error executing file {batch.file}', command=cmd) from e
        finally:
            # The clean-up runs to its end, even if the task is cancelled again meanwhile (a pool cancels its jobs on
            #  exit, after a `gather()` of them has already done so), since stopping the programs takes a moment
            await _uninterrupted(self._finish_batch(batch, stdout_task, stderr_task))

    async def _finish_batch(
        self, batch: BatchFile, stdout_task: asyncio.Task | None, stderr_task: asyncio.Task | None
    ) -> None:
        """
        Stop what's left of a batch file after it ran: the batch itself if it still runs, or the programs it started if
        it didn't complete (e.g. the shell ended, but a program it started still holds the output streams), then the
        tasks that copy its output.
        """
        if batch.process and (batch.process.returncode is None or batch.status != BatchFileStatus.COMPLETED):
            batch.logger.debug('Terminating batch file %(file)s', {'file': batch.file})
            await self._terminate_batch(batch)

        # Once the programs are gone, the streams end: let the logging tasks read what's left, so that the pipes close,
        #  then cancel any that still wait
        pending = [task for task in (stdout_task, stderr_task) if task and not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=1)
        for task in pending:
            if not task.done():
                batch.logger.debug('Cancelling async task %(name)s', {'name': task.get_name()})
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def cancel_batch(self, batch: BatchFile, **kwargs):
        if batch.status != BatchFileStatus.RUNNING:
            batch.logger.warning('Cannot cancel batch file %(file)s; it is not running', {'file': batch.file})
            return
        batch.logger.info('Cancelling batch file %(file)s', {'file': batch.file})
        batch._status = BatchFileStatus.CANCELLED
        await self._terminate_batch(batch)

    _TERMINATE_GRACE: float = 5.0
    """Seconds that a batch file's programs get to stop after each request, before a firmer one."""

    @staticmethod
    async def _terminate_batch(batch: BatchFile, **kwargs):
        """
        Stop a batch file and every program it started: on POSIX, its process group gets ``SIGINT`` (what Ctrl+C in a
        terminal sends), then ``SIGTERM``, then ``SIGKILL``, each after a grace period if something still runs; on
        Windows, its process group gets ``CTRL_BREAK_EVENT`` (Ctrl+C doesn't reach a group of its own), then, if the
        shell still runs after the grace period, ``taskkill /T /F`` ends it and its tree.
        """
        process = batch.process
        if process is None:
            return
        if OS_POSIX:
            await LocalSite._signal_group(batch)
        elif process.returncode is None:
            await LocalSite._break_group(batch)
            if process.returncode is None:
                await LocalSite._kill_tree(batch)

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    async def _signal_group(batch: BatchFile) -> None:
        # The shell leads a session of its own, so its process group is its pid
        pgid = batch.process.pid
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if not LocalSite._group_alive(pgid):
                break
            batch.logger.debug(
                'Sending %(signal)s to the processes of batch file %(file)s', {'signal': sig.name, 'file': batch.file}
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, sig)
            deadline = asyncio.get_running_loop().time() + LocalSite._TERMINATE_GRACE
            while LocalSite._group_alive(pgid) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
        await batch.process.wait()

    @staticmethod
    async def _break_group(batch: BatchFile) -> None:
        # The shell leads a process group of its own, so the group's id is its pid
        process = batch.process
        batch.logger.debug('Sending CTRL_BREAK_EVENT to the processes of batch file %(file)s', {'file': batch.file})
        try:
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
        except OSError:
            # E.g. no console to send it through: the tree is ended instead
            return
        deadline = asyncio.get_running_loop().time() + LocalSite._TERMINATE_GRACE
        while process.returncode is None and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

    @staticmethod
    async def _kill_tree(batch: BatchFile) -> None:
        pid = batch.process.pid
        batch.logger.debug('Ending the process tree of batch file %(file)s', {'file': batch.file})
        taskkill = await asyncio.create_subprocess_exec(
            'taskkill',
            '/T',
            '/F',
            '/PID',
            str(pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await taskkill.wait()
        if taskkill.returncode != 0 and batch.process.returncode is None:
            # E.g. the shell ended meanwhile, or taskkill isn't available: at least end the shell
            batch.process.kill()
        await batch.process.wait()

    @staticmethod
    async def _log_stream_to_file(stream: asyncio.StreamReader, file: Optional[Path] = None):
        # Open the file for writing if provided
        f = await aiofiles.open(file, mode='wb') if file else None
        try:
            while True:
                # Read 8 KB of data from the stream
                chunk = await stream.read(8192)

                # Break if no more data is available
                if not chunk:
                    break

                if f:
                    # Write the chunk to the file
                    await f.write(chunk)
                # Otherwise, the chunk is discarded, but still drained from the stream
        finally:
            # Close the file if it was opened
            if f:
                await f.close()


class ModulesSite(LocalSite):
    """
    A compute site that uses
    `Environment Modules <https://modules.readthedocs.io/en/latest/>`_ to manage
    software dependencies.
    """

    def _setup_lines(self, dependencies: list[DependencySettings], shell: ShellType, report: bool = False) -> list[str]:
        """
        On this site: the module command made available if the shell needs it, ``module purge``, the common modules
        and then each dependency's, in their order (one line per dependency, or per module with ``report``), then the
        directories put first on ``PATH`` (see :meth:`LocalSite.prepare_setup`).
        """
        from xtl import settings

        lines = [shell.module_setup] if shell.module_setup else []
        groups = [settings.dependencies.modules, *(dep.modules for dep in dependencies)]
        if report and any(groups):
            # Whether the shell has the module command, so that a missing one is told apart from missing modules
            lines.append(shell.get_found_report_command(shell.module_command, kind='command'))
        lines.append(shell.get_module_command('purge'))
        for modules in groups:
            if not modules:
                continue
            if report:
                lines.extend(shell.get_load_report_command(module) for module in modules)
            else:
                lines.append(shell.get_load_command(modules))
        return lines + super()._setup_lines(dependencies, shell, report=report)


class SchedulerSite(BaseComputeSite):
    """
    Abstract base class for compute sites that utilize a job scheduler system.
    """

    @abstractmethod
    async def schedule_batch(self, batch: BatchFile, **kwargs) -> str: ...

    @abstractmethod
    def query_status(self, job_id: str): ...

    @abstractmethod
    def cancel_job(self, job_id: str): ...


class SlurmSite(SchedulerSite, ABC):
    """
    Abstract base class for compute sites that utilize the
    `SLURM <https://slurm.schedmd.com/>`_ job scheduler.
    """

    _default_shell = Shell.BASH
    _supported_shells = frozenset([Shell.BASH])

    @staticmethod
    def _prepare_slurm_preamble(shell: ShellType, config: BatchJobConfig | dict = None) -> str:
        from xtl.jobs.config import BatchJobConfig

        args = []
        if config:
            if not isinstance(config, BatchJobConfig):
                config = BatchJobConfig(**config)
            args.extend(config.to_slurm())

        if args:
            nl = shell.new_line_char
            return nl.join(f'#SBATCH {arg}' for arg in args) + nl
        return ''

    def _prepare_directives(self, shell: ShellType, config: BatchJobConfig | dict = None, **kwargs) -> str:
        # The `#SBATCH` lines, which must come before any command
        return self._prepare_slurm_preamble(shell=shell, config=config) if config else ''

    async def validate_submission(self, batch: BatchFile, **kwargs) -> bool:
        # Run `sbatch --test-only <batchfile>` to check if allocation is possible
        #  and check the exit code
        raise NotImplementedError()

    async def schedule_batch(self, batch: BatchFile, **kwargs) -> str:
        # Run `sbatch <batchfile>` to submit the batch file
        raise NotImplementedError()

    def query_status(self, job_id: str):
        # Run `squeue --job <job_id>` to query the status of the batch file
        raise NotImplementedError()

    def cancel_job(self, job_id: str):
        # Run `scancel <job_id>` to cancel the batch file
        raise NotImplementedError()


class SlurmLocalSite(SlurmSite, LocalSite):
    """
    A compute site that utilizes the `SLURM <https://slurm.schedmd.com/>`_ job scheduler
    and finds executables on PATH, as :class:`LocalSite` does. Dependencies are checked on
    the submit host, and the guard lines of the batch file check them again on the
    compute node, where the job runs.
    """


class SlurmModulesSite(SlurmSite, ModulesSite):
    """
    A compute site that utilizes the `SLURM <https://slurm.schedmd.com/>`_ job scheduler
    and `Environment Modules <https://modules.readthedocs.io/en/latest/>`_ to manage
    software dependencies, as :class:`ModulesSite` does. Dependencies are checked on the
    submit host, and the guard lines of the batch file check them again on the compute
    node, where the job runs.
    """


class VirtualizationSite(BaseComputeSite, ABC): ...


class WSLSite(VirtualizationSite): ...


# NB: We could implement a new WSLComputeSite enum with tabulated values for specific
#  distros


class RemoteSite(BaseComputeSite, ABC): ...


class SSHSite(RemoteSite): ...


# Specialized compute sites (not general use)
class BiotixHPC(SlurmModulesSite):
    """
    A compute site that represents the Biotix HPC cluster.
    """

    _default_shell = Shell.BASH.get()
    _supported_shells = frozenset({Shell.BASH.get()})
    _policy = CommandPolicy.DEFAULT.get()


# Type alias
ComputeSiteType = LocalSite | ModulesSite | SchedulerSite
