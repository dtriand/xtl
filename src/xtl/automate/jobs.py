import asyncio
from functools import wraps
from pathlib import Path

from batchfile import BatchFile
from sites import ComputeSite, LocalSite


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
        wrapper.__is_semaphore_limited = True
        return wrapper
    return decorator



class Job:
    _no_parallel_jobs = 10
    __is_semaphore_modified = False

    def __init__(self, name: str, compute_site: ComputeSite = None, stdout_log: str | Path = None,
                 stderr_log: str | Path = None):
        self._job_type = 'job'
        self._name = str(name)

        if compute_site is None:
            compute_site = LocalSite()
        elif not isinstance(compute_site, ComputeSite):
            raise TypeError(f"compute_site must be an instance of ComputeSite, not {type(compute_site)}")
        self._compute_site = compute_site

        self._stdout = Path(stdout_log) if stdout_log is not None else Path(f'{self._name}.stdout.log')
        self._stderr = Path(stderr_log) if stderr_log is not None else Path(f'{self._name}.stderr.log')

    def create_batch(self, filename: str, cmds: list[str], do_chmod: bool = True):
        b = BatchFile(name=f'{self._name}', filename=filename, compute_site=self._compute_site)
        b.add_commands(*cmds)
        b.save(do_chmod=do_chmod)
        return b._filename

    async def run_batch(self, batchfile: str | Path, arguments: list[str] = None, stdout_log: str | Path = None,
                        stderr_log: str | Path = None):
        # Setup file streams for STDOUT and STDERR of the batch file
        if stdout_log is None:
            stdout_log = self._stdout
        else:
            stdout_log = Path(stdout_log)
        if stderr_log is None:
            stderr_log = self._stderr
        else:
            stderr_log = Path(stdout_log)

        # Create the log files if they don't exist
        stdout_log.touch(exist_ok=True)
        stderr_log.touch(exist_ok=True)

        # Create arguments list
        if arguments is None:
            arguments = list()

        # Run the batch file
        try:
            # Launch subprocess and capture STDOUT and STDERR
            #  This will run on a separate thread and/or core, determined by the underlying OS
            #  Once the subprocess is launched, the main thread will continue to the next line
            p = await asyncio.create_subprocess_exec(program=batchfile, *arguments,
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            # Log STDOUT and STDERR to files
            #  This keeps reading from PIPE until the subprocess exits and the buffer is empty
            await asyncio.gather(
                self._log_stream_to_file(p.stdout, stdout_log),
                self._log_stream_to_file(p.stderr, stderr_log)
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

    async def _log_stream_to_file(self, stream, log_file):
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
            __is_semaphore_modified = True

        # Get all methods of the subclass
        methods = [func for func in dir(SubJob) if callable(getattr(SubJob, func)) and not func.startswith("__")]

        # Find all methods that have been decorated with @limited_concurrency
        decorated = [func for func in methods if hasattr(getattr(SubJob, func), '__is_semaphore_limited')]

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
