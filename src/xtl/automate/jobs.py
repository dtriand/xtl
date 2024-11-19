import asyncio
from functools import wraps
from pathlib import Path
import warnings

from xtl.automate.shells import Shell, DefaultShell, BashShell, WslShell
from xtl.automate.sites import ComputeSite, LocalSite
from xtl.automate.batchfile import BatchFile


def limited_concurrency(limit: int):
    """
    Decorator to limit the number of concurrent executions of a function
    """
    semaphore = asyncio.Semaphore(limit)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Function is executed within a semaphore context
            async with semaphore:
                return await func(*args, **kwargs)
        # Tag the decorated function as semaphore-limited
        wrapper._is_semaphore_limited = True
        return wrapper
    return decorator



class Job:
    _no_parallel_jobs = 10
    _is_semaphore_modified = False
    _default_shell: Shell = BashShell
    _supported_shells = []
    _job_prefix = 'xtl.job'

    def __init__(self, name: str, compute_site: ComputeSite = None, shell: Shell = None, stdout_log: str | Path = None,
                 stderr_log: str | Path = None):
        """
        A class for executing jobs in the form of batch files.
        :param name: The name of the job - used for logging and identification
        :param compute_site: The ComputeSite where the job will be executed (default: LocalSite)
        :param shell: The Shell that will be used to execute the job (default: DefaultShell)
        :param stdout_log: The filename of the log file for STDOUT (default: <name>.stdout.log)
        :param stderr_log: The filename of the log file for STDERR (default: <name>.stderr.log)
        """
        self._job_type = 'xtl.job'
        self._name = str(name)

        # Set shell and compute_site
        self._shell, self._compute_site = self._determine_shell_and_site(shell, compute_site)

        # Set log files
        self._stdout = Path(stdout_log) if stdout_log is not None else Path(f'{self._name}.stdout.log')
        self._stderr = Path(stderr_log) if stderr_log is not None else Path(f'{self._name}.stderr.log')

        # Set echo function
        #  TODO: Implement a logging system
        self._echo = print

    def _determine_shell_and_site(self, shell: Shell = None, compute_site: ComputeSite = None):
        """
        Determine the shell and compute_site to use
        """
        if compute_site is None:  # Default compute site is LocalSite
            compute_site = LocalSite()
        elif not isinstance(compute_site, ComputeSite):
            raise TypeError(f'\'compute_site\' must be an instance of ComputeSite, not {type(compute_site)}')

        if shell is None:  # automatically determine the shell
            if not compute_site.supported_shells:  # if the compute site doesn't specify any requirements
                shell = self._default_shell
            elif not self._supported_shells:  # if the job doesn't specify any requirements
                shell = compute_site.default_shell
            else:
                common_shells = [s for s in self._supported_shells if s in compute_site.supported_shells]
                if self._default_shell in common_shells:  # if the default shell is supported by the site
                    shell = self._default_shell
                elif common_shells:  # else choose the first common shell
                    shell = common_shells[0]
            if shell is None:  # if still no shell is found
                shell = DefaultShell
        elif not isinstance(shell, Shell | WslShell):  # if shell is provided but not a Shell
            raise TypeError(f'\'shell\' must be an instance of Shell, not {type(shell)}')

        # Raise warnings for incompatible shells but continue
        if self._supported_shells:
            if not shell in self._supported_shells:
                warnings.warn(f'Shell \'{shell.name}\' is not compatible with job '
                              f'\'{self.__class__.__name__}\'')
        if not compute_site.is_valid_shell(shell):
            warnings.warn(f'Shell \'{shell.name}\' is not compatible with compute_site '
                          f'\'{compute_site.__class__.__name__}\'')
        return shell, compute_site

    def echo(self, message: str):
        """
        Echo a message to console using the specified echo function
        """
        self._echo(f'[{self._name}] {message}')

    def create_batch(self, filename: str | Path, cmds: list[str], change_permissions: bool = True) -> BatchFile:
        """
        Generate a batch file with the specified commands.

        :param filename: The name of the batch file
        :param cmds: A list of commands to be executed
        :param change_permissions: Whether to make the batch file executable (default: True)
        """
        b = BatchFile(filename=filename, compute_site=self._compute_site, shell=self._shell)
        b.add_commands(cmds)
        b.save(change_permissions=change_permissions)
        return b

    async def run_batch(self, batchfile: BatchFile, arguments: list[str] = None, stdout_log: str | Path = None,
                        stderr_log: str | Path = None):
        """
        Execute a batch file with the specified arguments

        :param batchfile: The BatchFile to execute
        :param arguments: Additional arguments to pass to the batch file
        :param stdout_log: The filename of the log file for STDOUT (default: <name>.stdout.log)
        :param stderr_log: The filename of the log file for STDERR (default: <name>.stderr.log)
        """
        # Setup log files
        stdout, stderr = self._setup_log_files(stdout_log, stderr_log)

        # Get the command and arguments to execute the batch file
        executable, arguments = self._get_executable_and_args(batchfile, arguments)

        # Run the batch file
        try:
            # Launch subprocess and capture STDOUT and STDERR
            #  This will run on a separate thread and/or core, determined by the underlying OS
            #  Once the subprocess is launched, the main thread will continue to the next line
            p = await asyncio.create_subprocess_exec(executable, *arguments, shell=False,
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            # Log STDOUT and STDERR to files
            #  This keeps reading from PIPE until the subprocess exits and the buffer is empty
            await asyncio.gather(
                self._log_stream_to_file(p.stdout, stdout),
                self._log_stream_to_file(p.stderr, stderr)
            )

        # If a SIGINT is received, terminate the subprocess and raise an exception
        except asyncio.CancelledError:
            p.terminate()
            raise Exception('Job cancelled by the user')

        # If the batch file doesn't exist or can't be launched
        except OSError as e:
            raise Exception(f'Failed to launch job:\n{e}')

        # Log the completion of the job
        finally:
            pass

    def _setup_log_files(self, stdout_log: str | Path, stderr_log: str | Path) -> tuple[Path, Path]:
        """
        Set up the log files for STDOUT and STDERR
        """
        # Setup file streams for STDOUT and STDERR of the batch file
        if stdout_log is None:
            stdout_log = self._stdout
        else:
            stdout_log = Path(stdout_log)
        if stderr_log is None:
            stderr_log = self._stderr
        else:
            stderr_log = Path(stderr_log)

        # Create the log files if they don't exist
        stdout_log.touch(exist_ok=True)
        stderr_log.touch(exist_ok=True)
        return stdout_log, stderr_log

    def _get_executable_and_args(self, batchfile: BatchFile, arguments: list[str] = None):
        """
        Get the command and arguments to execute the batch file
        """
        if not isinstance(batchfile, BatchFile):
            raise TypeError(f'\'batchfile\' must be an instance of BatchFile not {type(batchfile)}')

        batch_command = batchfile.get_execute_command(arguments=arguments, as_list=True)
        executable = batch_command[0]
        arguments = batch_command[1:]
        return executable, arguments


    async def _log_stream_to_file(self, stream, log_file):
        """
        Save a chunk of data from a stream to a log file
        """
        with open(log_file, "wb") as log:
            while True:
                # Read 4 KB of data from the stream
                buffer = await stream.read(1024 * 4)

                # If the buffer is empty, break the loop, i.e. the process has exited
                if not buffer:
                    break

                # Write the buffer to the log file
                log.write(buffer)

                # Flush the buffer to the log file
                log.flush()

    @staticmethod
    def save_to_file(filename: str, content: str):
        f = Path(filename)
        f.write_text(content, encoding='utf-8')
        return f

    @classmethod
    def update_concurrency_limit(cls, limit: int):
        """
        Create a new subclass of the current class and apply a new concurrency limit
        """

        # Create new subclass of self
        class SubJob(cls):
            # Update the concurrency limit
            _no_parallel_jobs = limit
            # Tag the subclass as modified (for debug)
            _is_semaphore_modified = True

        # Get all methods of the subclass
        methods = [func for func in dir(SubJob) if callable(getattr(SubJob, func)) and not func.startswith("__")]

        # Find all methods that have been decorated with @limited_concurrency
        decorated = [func for func in methods if hasattr(getattr(SubJob, func), '_is_semaphore_limited')]

        # Redecorate each of the methods with the new semaphore
        for method in decorated:
            decorated_method = getattr(SubJob, method)
            old_method = decorated_method.__wrapped__
            new_method = limited_concurrency(SubJob._no_parallel_jobs)(old_method)

            # Replace the old method with the new one
            setattr(SubJob, method, new_method)

        # Update __name__ and __qualname__ of the subclass
        SubJob.__name__ = f'Modified{cls.__name__}'
        SubJob.__qualname__ = f'Modified{cls.__qualname__}'
        return SubJob
