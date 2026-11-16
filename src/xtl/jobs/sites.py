from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import contextlib
import logging
from enum import Enum
import os
from pathlib import Path
import re
import signal
import subprocess
from typing import Optional, Iterable, Sequence, TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from xtl.jobs.config import BatchJobConfig
    from xtl.config.settings import DependencySettings
from xtl_core.system.compatibility import OS_POSIX, OS_WINDOWS, PY310_OR_LESS
from xtl.shared.environment import XTL_COMPUTE_SITE
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus
from xtl.jobs.policies import CommandPolicy, CommandPolicyType
from xtl.jobs.shells import Shell, ShellType
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

    def get(self) -> ComputeSiteType:
        mapping: dict[str, type] = {
            ComputeSite.LOCAL: LocalSite,
            ComputeSite.MODULES: ModulesSite,
        }
        if XTL_COMPUTE_SITE.value == 'BIOTIX_HPC':
            mapping[ComputeSite.BIOTIX] = BiotixHPC
        return mapping[self]()

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

    def __init__(self):
        pass

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

    def is_valid_shell(self, shell: Shell) -> bool:
        if not self.supported_shells:
            return True
        return shell in self.supported_shells

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
    ) -> bool: ...

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
            result = subprocess.run(command, capture_output=True, text=True, shell=False, timeout=timeout)
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
    A compute site that represents the local machine. It assumes that all required
    executables are available on PATH.
    """

    def prepare_preamble(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> str:
        preamble = self._prepare_shebang(shell)
        return self.policy.intercept_preamble(preamble) if self.policy else preamble

    def prepare_command(self, command: str) -> str:
        return self.policy.intercept_command(command) if self.policy else command

    def prepare_content(self, content: str) -> str:
        return self.policy.intercept_content(content) if self.policy else content

    def prepare_postamble(self) -> str:
        return self.policy.intercept_postamble('') if self.policy else ''

    def check_dependencies(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> bool:

        for dep in self.resolve_dependencies(dependencies):
            name = self._dependency_name(dep)
            # Check if the dependency executables are available
            for exe in dep.provides:
                try:
                    cmd = shell.get_dependency_resolution_command(dependency=exe, as_list=True)
                except ValueError as e:
                    # ValueError indicates an unsafe dependency name
                    logger.error(e)
                    return False

                result = self.execute_command(command=cmd, timeout=20)
                if result is None:
                    return False
                if result.stdout.strip() == '0':
                    logger.debug(
                        'Found executable `%(exe)s` for dependency `%(dependency)s`',
                        {'exe': exe, 'dependency': name},
                    )
                else:
                    logger.error(
                        'Missing executable `%(exe)s` for dependency `%(dependency)s`',
                        {'exe': exe, 'dependency': name},
                    )
                    return False
        else:
            logger.debug('All dependencies are met.')
            return True

    @staticmethod
    def _dependency_name(dependency: DependencySettings) -> str:
        """
        The name the settings know a dependency by, e.g. ``'autoproc'``, or its class name if they don't.
        """
        from xtl import settings

        return next(
            (name for name, known in settings.dependencies.items() if known is dependency), type(dependency).__name__
        )

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

    @staticmethod
    def _purge_modules(shell: ShellType) -> str:
        """
        Prepare the command to purge all loaded modules.

        :param shell: The shell type to generate the command for.
        :return: The command string to purge modules.
        """
        cmd = 'module purge'
        if shell == Shell.CMD:
            # For Windows CMD, we need to use 'call' to execute the module commands
            cmd = f'call {cmd}'
        return cmd + shell.new_line_char

    @staticmethod
    def _load_modules(modules: str | Iterable[str], shell: ShellType) -> str:
        """
        Prepare the command to load specified modules.

        :param modules: An iterable of module names to load.
        :param shell: The shell type to generate the command for.
        :return: The command string to load the modules.
        """
        if isinstance(modules, str):
            modules = [modules]
        cmd = f'module load {" ".join(modules)}'
        if shell == Shell.CMD:
            # For Windows CMD, we need to use 'call' to execute the module commands
            cmd = f'call {cmd}'
        return cmd + shell.new_line_char

    def _prepare_modules_preamble(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> str:
        from xtl import settings

        preamble = self._purge_modules(shell=shell)
        if common := settings.dependencies.modules:
            # Modules that every job loads, before those of its dependencies
            preamble += self._load_modules(modules=common, shell=shell)
        for dep in self.resolve_dependencies(dependencies):
            if dep.modules:
                preamble += self._load_modules(modules=dep.modules, shell=shell)
        return preamble

    def prepare_preamble(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> str:
        """
        Prepare the preamble for loading required modules based on dependencies. The
        preamble has the following structure:

        .. code-block:: text
            module purge
            module load <module1> <module2> ...

        :param dependencies: An iterable of dependency names (str) or DependencySettings
            instances, or None.
        :param shell: The shell type to generate the commands for.
        :return: The preamble string.
        """
        preamble = self._prepare_shebang(shell)
        preamble += self._prepare_modules_preamble(dependencies=dependencies, shell=shell)
        return self.policy.intercept_preamble(preamble) if self.policy else preamble

    def check_dependencies(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None, shell: ShellType
    ) -> bool:
        # Check if the `module` command is available
        cmd = shell.get_dependency_resolution_command(dependency='module', as_list=True)
        result = self.execute_command(command=cmd, timeout=20)
        if result is None:
            return False
        if result.stdout.strip() == '0':
            logger.debug('Found `module` command for dependency resolution')
        else:
            logger.error('Missing `module` command for dependency resolution')
            return False

        # Check the modules that every job loads, then those of each dependency
        from xtl import settings

        modules = list(settings.dependencies.modules)
        for dep in self.resolve_dependencies(dependencies):
            modules.extend(dep.modules)
        if not all(self._check_module(module, shell=shell) for module in modules):
            return False
        logger.debug('All dependencies are met.')
        return True

    def _check_module(self, module: str, shell: ShellType) -> bool:
        """
        Check that a module is available, with ``module avail -t -a <module>``.

        :param module: The module name.
        :param shell: The shell to run the command in.
        :return: True if the module is available, False otherwise (also for an unsafe name).
        """
        module = str(module).strip()
        if not self._is_module_name_safe(module):
            logger.error(
                'Invalid module name `%(module)s`. Module names may '
                'only contain alphanumeric characters, hyphens, '
                'underscores, periods and slashes.',
                {'module': module},
            )
            return False
        cmd = shell.get_execute_command(f'module avail -t -a {module}', as_list=True)
        result = self.execute_command(command=cmd, timeout=20)
        if result is None:
            return False
        if module in result.stdout:
            logger.debug('Found module `%(module)s`', {'module': module})
            return True
        logger.error('Missing module `%(module)s`', {'module': module})
        return False

    @staticmethod
    def _is_module_name_safe(module: str) -> bool:
        """
        Check if the module name is safe to use in shell commands. A safe module name
        contains only alphanumeric characters, hyphens, underscores, periods and
        slashes, and no part of it starts with a hyphen (which would read as an option).

        :param module: The module name to check.
        :return: True if the module name is safe, False otherwise.
        """
        # Check length constraints
        if len(module) == 0:
            return False
        elif len(module) > 255:
            # Prevent potential buffer overflow attacks
            return False
        # Pattern for a safe module name: slash separated segments of alphanumeric, dot, dash and underscore
        #  characters, none starting with a dash
        segment = r'[A-Za-z0-9_.][A-Za-z0-9_.-]*'
        return re.fullmatch(rf'{segment}([/\\]{segment})*', module) is not None


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
    and assumes that all required executables are available on PATH.
    """

    def prepare_preamble(
        self,
        dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None,
        shell: ShellType,
        config: BatchJobConfig | dict = None,
        **kwargs,
    ) -> str:
        preamble = self._prepare_shebang(shell=shell)
        # SBATCH preamble
        if config:
            preamble += self._prepare_slurm_preamble(shell=shell, config=config)
        return self.policy.intercept_preamble(preamble) if self.policy else preamble


class SlurmModulesSite(SlurmSite, ModulesSite):
    """
    A compute site that utilizes the `SLURM <https://slurm.schedmd.com/>`_ job scheduler
    and `Environment Modules <https://modules.readthedocs.io/en/latest/>`_ to manage
    software dependencies.
    """

    def prepare_preamble(
        self,
        dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None,
        shell: ShellType,
        config: BatchJobConfig | dict = None,
        **kwargs,
    ) -> str:
        preamble = self._prepare_shebang(shell=shell)
        # SBATCH preamble
        if config:
            preamble += self._prepare_slurm_preamble(shell=shell, config=config)
        # Modules preamble
        preamble += self._prepare_modules_preamble(dependencies=dependencies, shell=shell)
        return self.policy.intercept_preamble(preamble) if self.policy else preamble


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
