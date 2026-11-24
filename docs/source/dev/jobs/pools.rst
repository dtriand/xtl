.. |SimplePool| replace:: :class:`SimplePool <xtl.jobs.pools.SimplePool>`
.. |AsyncPool| replace:: :class:`AsyncPool <xtl.jobs.pools.AsyncPool>`
.. |ThreadedPool| replace:: :class:`ThreadedPool <xtl.jobs.pools.ThreadedPool>`
.. |MultiprocessPool| replace:: :class:`MultiprocessPool <xtl.jobs.pools.MultiprocessPool>`
.. |Resources| replace:: :class:`Resources <xtl.jobs.resources.Resources>`
.. |JobResults| replace:: :class:`JobResults <xtl.jobs.results.JobResults>`
.. |JobStatus| replace:: :class:`JobStatus <xtl.jobs.results.JobStatus>`

Pools
=====

A pool runs many jobs, at most ``max_jobs`` at a time. All pools have the same interface, so any job runs in any of
them; they differ in where the jobs run.

The pool types
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Pool
     - Runs the jobs
     - Best for
   * - |SimplePool|
     - in the event loop, one after the other
     - running jobs strictly in order, e.g. while debugging
   * - |AsyncPool|
     - in the event loop, concurrently
     - batch jobs, and anything that mostly waits (e.g. API calls)
   * - |ThreadedPool|
     - in worker threads
     - Python code that blocks but releases the GIL (I/O, numpy, C extensions)
   * - |MultiprocessPool|
     - in worker processes
     - CPU-bound Python code; slower to start

Thread and process pools build a copy of each job in the worker, which brings a few rules
(:ref:`jobs-in-workers`).

Where a pool type is configurable, it goes by name: ``'simple'``, ``'async'``, ``'threads'`` or ``'processes'``
(``JobPool('threads').get()`` returns |ThreadedPool|).

Running jobs
------------

Running jobs in a pool takes three steps: open the pool, submit the jobs, and launch them.

.. code-block:: python

    from xtl.jobs import ThreadedPool

    configs = [StatsConfig(values=v) for v in ([1, 2, 3], [2, 4, 6], [], [5, 5])]

    async with ThreadedPool(max_jobs=2, name='stats') as pool:  # 1. open
        jobs = pool.submit(StatsJob, configs=configs)           # 2. submit
        results = await pool.launch()                           # 3. launch

    [job.job_id for job in jobs]  # ['StatsJob-1', 'StatsJob-2', 'StatsJob-3', 'StatsJob-4']
    [r.success for r in results]  # [True, True, False, True]

#. **Open** the pool with ``async with``. Its arguments are ``max_jobs``, how many jobs run at once, and optionally a
   ``name``, shown in its logs, and a ``logger_config`` (:doc:`logging`). A pool is single-use: once it has exited,
   entering it again raises :class:`RuntimeError`, so create a new one for the next batch.
#. **Submit** jobs with :meth:`~xtl.jobs.pools.BasePool.submit`: a job class and a list of configurations, one per job.
   The pool numbers the jobs (``StatsJob-1``, ``StatsJob-2``...) and returns them. Nothing runs yet.
#. **Launch** them with :meth:`~xtl.jobs.pools.BasePool.launch`, which runs the jobs submitted so far and returns
   their |JobResults| in the order they were submitted. To handle each result as soon as its job finishes instead,
   stream them:

   .. code-block:: python

       async for result in await pool.launch('stream'):
           print(result.job_id, result.success)

   You can submit and launch again within the same ``async with``; each launch runs only the jobs submitted since the
   previous one.

**Dependencies:** before a launch runs its jobs, the pool checks the external programs they need (``_dependencies``,
see :doc:`batch-jobs`), once for all of them: each dependency is checked once per compute site and shell, and the
outcome is remembered, so the jobs don't check again when they start (in a process pool, the workers receive it with
the settings). A job whose dependency is missing doesn't run: its result has a
:class:`~xtl.jobs.exceptions.DependencyError`, its status is ``FAILED``, and the other jobs run as usual. Nothing is
checked if ``dependencies.check`` is off in ``xtl.toml``.

**When something fails:** a failing job doesn't stop the others; its exception is in its result (above, the empty list
in ``StatsJob-3``). An exception in your own code inside the ``async with`` block, ``Ctrl+C`` or a cancellation is
different: the pool stops the running jobs (see `Cancellation`_), and the exception propagates.

Job status
----------

While the jobs run, the pool knows where each one stands, as a |JobStatus|: ``SUBMITTED``, ``QUEUED`` (launched,
waiting for a slot), ``RUNNING``, ``DONE``, ``FAILED`` or ``CANCELLED``:

.. code-block:: python

    async with AsyncPool(max_jobs=2) as pool:
        pool.submit(StatsJob, configs=configs)
        task = asyncio.create_task(pool.launch())
        ...
        pool.counts()  # {JobStatus.QUEUED: 2, JobStatus.RUNNING: 2, JobStatus.DONE: 0, ...}
        pool.status    # {'StatsJob-1': JobStatus.RUNNING, 'StatsJob-2': JobStatus.RUNNING, 'StatsJob-3': ...}

Both return copies, which stay readable after the pool exits, e.g. to draw a progress bar from another thread.

Cancellation
------------

A pool cancels its running and waiting jobs when:

- your code inside the ``async with`` block raises an exception
- ``Ctrl+C`` is pressed
- the task that awaits ``launch()`` is cancelled, e.g. by a timeout:

.. code-block:: python

    async with ThreadedPool(max_jobs=2) as pool:
        pool.submit(SlowJob, configs=configs)
        results = await asyncio.wait_for(pool.launch(), timeout=60)  # TimeoutError after 60 s

The exception (here :class:`TimeoutError`) propagates once the jobs have stopped, and there are no results; the
pool's ``status`` shows each job as ``CANCELLED``.

**What a job sees:** :class:`asyncio.CancelledError`, raised at the ``await`` where it waits, in every pool type (in a
thread or process pool, within a fraction of a second). Clean up in a ``finally`` block:

.. code-block:: python

    class SlowJob(Job[JobConfig]):
        async def _execute(self):
            try:
                await asyncio.sleep(600)
            finally:
                self.logger.info('cleaning up')  # runs when the job is cancelled too

A batch job stops the programs it started (:doc:`batch-jobs`).

**A job that doesn't stop**, because it catches the cancellation, or blocks without awaiting (e.g. ``time.sleep()``, a
long computation), gets a grace period of 15 seconds. Then a process pool ends its worker processes, while a thread
pool, which can't end a thread, waits for it.

If a worker process dies (a crash, or the system killing it for memory), the process pool is broken: the jobs running
at the time, and every job launched later in that pool, fail with
:class:`~concurrent.futures.process.BrokenProcessPool`.

Resources
---------

Pools share a machine's resources: a count of ``jobs``, ``threads`` and ``processes`` (a |Resources|). The totals come
from the ``[jobs.resources]`` section of ``xtl.toml`` (by default, the number of CPU cores each).

- A pool takes its share when it's entered and gives it back on exit. If not enough is free, entering waits for
  another pool to give some back.
- A job can ask for less than its pool's share: its configuration's ``resources``, or else its class's
  ``_resources``.
- A pool opened inside a job (:meth:`~xtl.jobs.jobs.Job.get_pool`) gets its share from that job's, so nested pools
  never use more than their parent. ``get_pool()`` lowers ``max_jobs`` to what's available, with a warning.

Sharing state between jobs
--------------------------

Locks, queues and states that the jobs use must be created with the pool before launching (see :doc:`jobs` for
using them inside a job):

.. code-block:: python

    async with ThreadedPool(max_jobs=8) as pool:
        state = pool.get_state('counts')
        state['n'] = 0
        pool.submit(Counter, configs=[None] * 32)  # each job adds 1, under the pool's default lock
        await pool.launch()
        state['n']  # 32

Values in a ``MultiprocessPool``'s queues and states cross processes, so they must be picklable.

Pools with a live display: ``LivePool``
---------------------------------------

The command line tools use ``xtl.tui.live_pool.LivePool``, which wraps a pool and draws a progress bar and the jobs'
latest log lines on the console:

.. code-block:: python

    from xtl.tui.console import ConsoleIO

    console = ConsoleIO(verbose=0, debug=False)
    async with console.get_pool('async', max_jobs=4) as pool:
        pool.submit(StatsJob, configs=configs)
        results = await pool.launch()

It has the same ``submit()``, ``launch()`` and shared-state methods. In addition, every job writes ``xtl.log`` in its
job directory (unless ``cli.job_log_files = false`` in ``xtl.toml``), and if a job fails, it offers to print all the
jobs' logs once the pool has exited.

A command line tool declares the external programs its jobs need with ``@requires``, and checks them once, before it
starts any jobs, with ``console.check_dependencies()``; its pools and jobs then find the outcome and don't check again.
If one is missing, the tool prints what's missing (and what the check printed, with ``-v``) and exits with code 1:

.. code-block:: python

    from xtl.cli.utilities.decorators import attach_hook, requires, typer_async


    @app.command('compare')
    @requires('atsas')
    @attach_hook(func=get_console_options, hook_output='console_options')
    @attach_hook(func=get_job_options, hook_output='job_options')
    @typer_async
    async def cli_saxs_compare(..., job_options: JobOptions = typer.Option(), console_options: ...):
        console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
        console.apply_job_options(job_options)  # --compute-site and --modules, into the settings
        console.check_dependencies()            # after the options, which may change where they're found
        async with console.get_pool('async', max_jobs=4) as pool:
            ...

Each problem gets a line of its own, which says what is missing and what needs it, *e.g.*:

.. code-block:: text

    Missing dependencies:
    - module atsas/9.9 not found (required for atsas)
    - executables datcmp, datop not found (required for atsas)
    Run with -v to see what the checks printed.

Other lines read ``module command `envmodule` not found (required for atsas, to load atsas/4.1.3)``, ``module gcc/12
not found (loaded for every job)``, ``atsas could not be checked: <reason>`` (*e.g.* the check didn't finish, or an
unsafe module name) or ``unknown dependency <name>``. A job's :class:`~xtl.jobs.exceptions.DependencyError` uses the
same lines.
