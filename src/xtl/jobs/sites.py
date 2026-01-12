from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from enum import Enum
from pathlib import Path
import re
import subprocess
from typing import Optional, Iterable, Sequence

import aiofiles

from xtl import settings
from xtl.config.settings import DependencySettings
from xtl.common.compatibility import PY310_OR_LESS, XTL_COMPUTE_SITE
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus
from xtl.jobs.policies import CommandPolicy, CommandPolicyType
from xtl.jobs.shells import Shell, ShellType
from xtl.logging import Logger


__all__ = ['ComputeSite', 'ComputeSiteType', 'BaseComputeSite', 'LocalSite',
           'ModulesSite']


logger = Logger(__name__)


if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class ComputeSite(StrEnum):
    # General purpose compute sites
    LOCAL = 'local'
    MODULES = 'modules'
    SLURM = 'slurm'
    SLURM_LOCAL = 'slurm_local'

    # Specialized compute sites
    if XTL_COMPUTE_SITE == 'BIOTIX_HPC':
        BIOTIX = 'biotix'

    def get(self) -> ComputeSiteType:
        mapping: dict[str, type] = {
            ComputeSite.LOCAL: LocalSite,
            ComputeSite.MODULES: ModulesSite,
            ComputeSite.SLURM: SlurmModulesSite,
            ComputeSite.SLURM_LOCAL: SlurmLocalSite,
        }
        if XTL_COMPUTE_SITE == 'BIOTIX_HPC':
            mapping[ComputeSite.BIOTIX] = BiotixHPC
        return mapping[self]()

    def __eq__(self, other):
        if isinstance(other, BaseComputeSite):
            return self.get() == other
        return super().__eq__(other)


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
            return (self.default_shell == other.default_shell and
                    self.supported_shells == other.supported_shells and
                    self.policy == other.policy)
        raise NotImplementedError(f'Cannot compare {type(self)} with {type(other)}')

    def is_valid_shell(self, shell: Shell) -> bool:
        if not self.supported_shells:
            return True
        return shell in self.supported_shells

    @abstractmethod
    def prepare_preamble(self, dependencies: Iterable[DependencySettings | str]
                                             | DependencySettings | str
                                             | None,
                         shell: ShellType) -> str: ...

    @abstractmethod
    def prepare_command(self, command: str) -> str: ...

    @abstractmethod
    def prepare_content(self, content: str) -> str: ...

    @abstractmethod
    def prepare_postamble(self) -> str: ...

    @staticmethod
    def resolve_dependencies(dependencies: Iterable[DependencySettings | str]
                                           | DependencySettings | str
                                           | None) -> set[DependencySettings]:
        """
        Resolves dependencies from settings based on the provided dependency names or
        DependencySettings objects.

        :param dependencies: An iterable of dependency names (str) or DependencySettings
            instances, or None.
        :return: A set of resolved DependencySettings instances.
        :raises ValueError: If a dependency name is not found in settings and strict
            resolution is enabled.
        """
        resolved_deps: set[DependencySettings] = set()
        if dependencies is None:
            return resolved_deps
        if isinstance(dependencies, (str, DependencySettings)):
            dependencies = [dependencies]

        for depname in dependencies:
            # Resolve dependency from settings
            if isinstance(depname, str):
                dep = getattr(settings.dependencies, depname, None)
            else:
                dep = depname

            # Check if dependency was found in settings
            if dep is None:
                logger.debug('Dependency `%s` not found in settings', depname)
                if settings.dependencies.resolution == 'strict':
                    logger.error('Strict dependency resolution enabled; missing '
                                 'dependency `%s`', depname)
                    raise ValueError(f'Missing dependency `{depname}`')
                continue

            resolved_deps.add(dep)
        return resolved_deps

    @abstractmethod
    def check_dependencies(self, dependencies: Iterable[DependencySettings | str]
                                               | DependencySettings | str
                                               | None,
                           shell: ShellType) -> bool: ...

    @staticmethod
    def execute_command(command: Sequence[str], timeout: int = 20) \
            -> subprocess.CompletedProcess | None:
        """
        Execute a command locally and return the CompletedProcess object.

        :param command: The command to execute as a list of strings.
        :param timeout: The timeout in seconds for the command execution.
        :return: The CompletedProcess object if the command executed successfully,
            None if the command timed out.
        """
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    shell=False, timeout=timeout)
        except TimeoutError:
            logger.error('Command `%s` timed out', ' '.join(command))
            return None
        return result

    @abstractmethod
    async def execute_batch(self, batch: BatchFile, **kwargs):
        ...


class LocalSite(BaseComputeSite):
    """
    A compute site that represents the local machine. It assumes that all required
    executables are available on PATH.
    """

    def prepare_preamble(self, dependencies: Iterable[DependencySettings | str]
                                             | DependencySettings | str
                                             | None,
                         shell: ShellType) -> str:
        return self.policy.intercept_preamble('') if self.policy else ''

    def prepare_command(self, command: str) -> str:
        return self.policy.intercept_command(command) if self.policy else command

    def prepare_content(self, content: str) -> str:
        return self.policy.intercept_content(content) if self.policy else content

    def prepare_postamble(self) -> str:
        return self.policy.intercept_postamble('') if self.policy else ''

    def check_dependencies(self, dependencies: Iterable[DependencySettings | str]
                                               | DependencySettings | str
                                               | None,
                           shell: ShellType) -> bool:

        for dep in self.resolve_dependencies(dependencies):
            # Check if the dependency executables are available
            for exe in dep.provides:
                try:
                    cmd = shell.get_dependency_resolution_command(dependency=exe,
                                                                  as_list=True)
                except ValueError as e:
                    # ValueError indicates an unsafe dependency name
                    logger.error(e)
                    return False

                result = self.execute_command(command=cmd, timeout=20)
                if result is None:
                    return False
                if result.stdout.strip() == '0':
                    logger.debug('Found executable `%(exe)s` for dependency '
                                 '`%(dependency)s`', {'exe': exe,
                                                      'dependency': dep.name})
                else:
                    logger.error('Missing executable `%(exe)s` for dependency '
                                 '`%(dependency)s`', {'exe': exe,
                                                      'dependency': dep.name})
                    return False
        else:
            logger.debug('All dependencies are met.')
            return True

    async def execute_batch(self, batch: BatchFile, stdout: Path = None,
                            stderr: Path = None, **kwargs):
        # Get command to execute the batch file
        cmd = batch.shell.get_execute_batch_command(batch.file)

        if PY310_OR_LESS:
            # TODO: Update exception handling to asyncio.TaskGroup when we drop support
            #  for Python 3.10
            pass

        # Initialize stream logging tasks
        stdout_task, stderr_task = None, None
        try:
            # Start batch execution
            batch._process = await asyncio.create_subprocess_exec(
                *cmd,
                shell=False,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            batch._status = BatchFileStatus.RUNNING

            # Start logging of streams
            stdout_task = asyncio.create_task(
                self._log_stream_to_file(batch.process.stdout, stdout),
                name='stdout_logger'
            )
            stderr_task = asyncio.create_task(
                self._log_stream_to_file(batch.process.stderr, stderr),
                name='stderr_logger'
            )

            # Wait for all tasks to complete. This is where the main thread waits.
            await asyncio.gather(batch.process.wait(), stdout_task, stderr_task)

            # Check if the batch was cancelled by .cancel_batch()
            if batch.status != BatchFileStatus.CANCELLED:
                batch._status = BatchFileStatus.COMPLETED
        except asyncio.CancelledError as e:
            # Raise cancellation up the stack after marking the batch as cancelled
            logger.error('Batch execution was cancelled by user')
            batch._status = BatchFileStatus.CANCELLED
            raise e
        except Exception as e:
            # Log any other exceptions during execution
            logger.error('Error executing batch file %{file}s: %{exc}s',
                         {'file': batch.file, 'exc': str(e)})
            batch._status = BatchFileStatus.FAILED
        finally:
            # Terminate batch if still running
            if batch.process and batch.process.returncode is None:
                logger.debug('Terminating batch file %{file}s',
                             {'file': batch.file})
                await self._terminate_batch(batch)

            # Terminate logging tasks
            for task in (stdout_task, stderr_task):
                if task and not task.done():
                    logger.debug('Cancelling async task %{name}s',
                                 {'name': task.get_name()})
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def cancel_batch(self, batch: BatchFile, **kwargs):
        if batch.status != BatchFileStatus.RUNNING:
            logger.warning('Cannot cancel batch file %{file}s; it is not running',
                           {'file': batch.file})
            return
        logger.info('Cancelling batch file %{file}s', {'file': batch.file})
        batch._status = BatchFileStatus.CANCELLED
        await self._terminate_batch(batch)

    @staticmethod
    async def _terminate_batch(batch: BatchFile, **kwargs):
        if batch.process and batch.process.returncode is None:
            # Request graceful termination of the process
            logger.debug('Sending SIGTERM to batch process PID %{pid}d',
                         {'pid': batch.process.pid})
            batch.process.terminate()
            try:
                # Wait 10 s for a graceful shutdown
                await asyncio.wait_for(batch.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                # Force kill the process
                logger.debug('Sending SIGKILL to batch process PID %{pid}d',
                             {'pid': batch.process.pid})
                batch.process.kill()
                await batch.process.wait()

    @staticmethod
    async def _log_stream_to_file(stream: asyncio.StreamReader,
                                  file: Optional[Path] = None):
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
        cmd = 'module purge' + shell.new_line_char
        if shell == Shell.CMD:
            # For Windows CMD, we need to use 'call' to execute the module commands
            cmd = f'call {cmd}'
        return cmd

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
        cmd = f'module load {" ".join(modules)}{shell.new_line_char}'
        if shell == Shell.CMD:
            # For Windows CMD, we need to use 'call' to execute the module commands
            cmd = f'call {cmd}'
        return cmd

    def prepare_preamble(self, dependencies: Iterable[DependencySettings | str]
                                             | DependencySettings | str
                                             | None,
                         shell: ShellType) -> str:
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
        preamble = self._purge_modules(shell=shell)
        for dep in self.resolve_dependencies(dependencies):
            if dep.modules:
                preamble += self._load_modules(modules=dep.modules, shell=shell)
        return self.policy.intercept_preamble(preamble) if self.policy else preamble

    def check_dependencies(self, dependencies: Iterable[DependencySettings | str]
                                               | DependencySettings | str
                                               | None,
                           shell: ShellType) -> bool:
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

        # Run `module avail -t -a <module>` to check if modules are available
        for dep in self.resolve_dependencies(dependencies):
            if not dep.modules:
                continue
            for module in dep.modules:
                module = str(module).strip()
                if not self._is_module_name_safe(module):
                    logger.error('Invalid module name `%(module)s`. Module names may '
                                 'only contain alphanumeric characters, hyphens, '
                                 'underscores, periods and slashes.',
                                 {'module': module})
                    return False
                cmd = shell.get_execute_command(f'module avail -t -a {module}',
                                                as_list=True)
                result = self.execute_command(command=cmd, timeout=20)
                if result is None:
                    return False
                if module in result.stdout:
                    logger.debug('Found module `%(module)s`', {'module': module})
                else:
                    logger.error('Missing module `%(module)s`', {'module': module})
                    return False
        else:
            logger.debug('All dependencies are met.')
            return True

    @staticmethod
    def _is_module_name_safe(module: str) -> bool:
        """
        Check if the module name is safe to use in shell commands. A safe module name
        contains only alphanumeric characters, hyphens, underscores, periods and
        slashes.

        :param module: The module name to check.
        :return: True if the module name is safe, False otherwise.
        """
        # Check length constraints
        if len(module) == 0:
            return False
        elif len(module) > 255:
            # Prevent potential buffer overflow attacks
            return False
        # Pattern for a safe module name (slash separated segments of alphanumeric,
        #  dash, and underscore characters plus optional leading dot)
        pattern = r'^\.?[A-Za-z0-9_-]+([/\\]\.?[A-Za-z0-9_-]+)*$'
        return re.match(pattern, module) is not None


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

    def prepare_preamble(self, dependencies: Iterable[DependencySettings | str]
                                             | DependencySettings | str
                                             | None,
                         shell: ShellType) -> str:
        # SBATCH preamble
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

    def prepare_preamble(self, dependencies: Iterable[DependencySettings | str]
                                             | DependencySettings | str
                                             | None,
                         shell: ShellType) -> str:
        slurm_preamble = SlurmSite.prepare_preamble(self, dependencies=dependencies,
                                                    shell=shell)
        local_preamble = LocalSite.prepare_preamble(self, dependencies=dependencies,
                                                    shell=shell)
        raise NotImplementedError()


class SlurmModulesSite(SlurmSite, ModulesSite):
    """
    A compute site that utilizes the `SLURM <https://slurm.schedmd.com/>`_ job scheduler
    and `Environment Modules <https://modules.readthedocs.io/en/latest/>`_ to manage
    software dependencies.
    """

    def prepare_preamble(self, dependencies: Iterable[DependencySettings | str]
                                             | DependencySettings | str
                                             | None,
                         shell: ShellType) -> str:
        slurm_preamble = SlurmSite.prepare_preamble(self, dependencies=dependencies,
                                                    shell=shell)
        modules_preamble = ModulesSite.prepare_preamble(self, dependencies=dependencies,
                                                        shell=shell)
        raise NotImplementedError()


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
