.. |BatchJob| replace:: :class:`BatchJob <xtl.jobs.jobs.BatchJob>`
.. |Job| replace:: :class:`Job <xtl.jobs.jobs.Job>`
.. |JobConfig| replace:: :class:`JobConfig <xtl.jobs.config.JobConfig>`
.. |JobResults| replace:: :class:`JobResults <xtl.jobs.results.JobResults>`
.. |Options| replace:: :class:`Options <xtl_core.options.base.Options>`
.. |run| replace:: :meth:`run() <xtl.jobs.jobs.Job.run>`
.. |SteppedJob| replace:: :class:`SteppedJob <xtl.jobs.jobs.SteppedJob>`

Writing jobs
============

A job is one unit of work, made of two classes: a configuration (|JobConfig|), which holds what the job needs to know,
and the job itself (|Job|), whose ``_execute()`` coroutine does the work. This page shows how to write both, and how
a job runs alone or in a pool.

Two kinds of job need no ``_execute()`` of their own:

- a |BatchJob| runs an external program through a script it generates from templates (see :doc:`batch-jobs`);
- a |SteppedJob| runs other jobs one after the other, each configured from what the earlier ones produced (see
  :doc:`stepped-jobs`).

Job and configuration
---------------------

A configuration is a subclass of |JobConfig|, which is an |Options| model (see :doc:`../../core/options`). Its
fields are typed and validated when the configuration is created, so a wrong value fails there, with a clear message,
rather than halfway through the job. A configuration also saves to and loads from a dict, JSON or TOML, so a job can
be described in a file, or sent to another process.

This job computes the mean and the standard deviation of a list of numbers:

.. code-block:: python

    import math

    from xtl_core.options import Option
    from xtl.jobs import Job, JobConfig


    class StatsConfig(JobConfig):
        values: list[float] = Option(..., desc='The numbers to summarize')


    class StatsJob(Job[StatsConfig]):
        async def _execute(self) -> dict[str, float]:
            values = self.config.values
            mean = sum(values) / len(values)
            stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            self.logger.info('%d values: mean %.2f, standard deviation %.2f', len(values), mean, stdev)
            return {'mean': mean, 'stdev': stdev}

A few things to note:

* The job names its configuration class in brackets, ``Job[StatsConfig]``, and accepts only configurations of that
  class.
* ``_execute()`` is a coroutine (``async def``), but it can run ordinary blocking code too; see :doc:`pools` for which
  pools run such jobs in parallel.
* Inside ``_execute()``, ``self.config`` is the configuration and ``self.logger`` the job's logger. Log with it
  instead of ``print()``, so that messages carry the job's name and reach the pool's handlers.
* What ``_execute()`` returns becomes the job's result.

Every |JobConfig| also has these fields:

- ``job_directory`` (default: ``None``): where the job works (see `Job directory`_);
- ``resources`` (default: ``None``): the most resources the job may use in a pool (see :doc:`pools`);
- ``log_handlers`` (default: empty): log handlers of the job's own (see :doc:`logging`).

Running a job
-------------

Run a single job directly, by awaiting its |run|:

.. code-block:: python

    job = StatsJob(StatsConfig(values=[2, 4, 4, 4, 5, 5, 7, 9]))
    result = await job.run()
    result.success  # True
    result.data     # {'mean': 5.0, 'stdev': 2.0}

A job runs once; running it again returns ``None``.

To run many jobs, submit them to a pool, which runs up to ``max_jobs`` of them at a time and returns their results
in the order they were submitted (see :doc:`pools`):

.. code-block:: python

    from xtl.jobs import AsyncPool

    async with AsyncPool(max_jobs=4) as pool:
        pool.submit(StatsJob, configs=[StatsConfig(values=v) for v in ([1, 2, 3], [2, 4, 4, 4, 5, 5, 7, 9], [])])
        results = await pool.launch()

    [r.data for r in results]
    # [{'mean': 2.0, 'stdev': 0.816...}, {'mean': 5.0, 'stdev': 2.0}, None]

Either way, each job gives a |JobResults|:

- ``job_id``: the job's id;
- ``data``: what ``_execute()`` returned, or ``None`` if it failed;
- ``error``: the exception that made the job fail (above, a :class:`ZeroDivisionError` for the empty list), or
  ``None``;
- ``success``: ``True`` if there is no error.

A stepped job's results also have the results of each step (see :doc:`stepped-jobs`).

Errors
------

``_execute()`` may raise any exception. The job catches it, logs it, and returns it as ``result.error``, so a failing
job never stops its caller or the other jobs of its pool. Only cancellation is raised again, so that it reaches the
caller.

Two errors come from the jobs machinery itself, not from job code:

- :class:`~xtl.jobs.exceptions.SubprocessError`: a batch job's program failed (a non-zero return code) or couldn't
  start; it holds the ``command`` that was run (see :doc:`batch-jobs`).
- :class:`~xtl.jobs.exceptions.WorkerError`: in a process pool, a job's error or data couldn't be sent back to the
  main process, because it can't be pickled. It holds the original error's type and traceback as text. An exception
  class of your own avoids it by being importable and picklable.

Job directory
-------------

A job works in its ``job_directory``, if its configuration has one. A batch job always has one: a new path in the
temporary directory unless the configuration gives another. Other jobs have none unless the configuration sets it.
Building, copying or pickling a configuration never touches the disk. Jobs that work in their directory, batch jobs
and stepped jobs (for their steps' directories), create it with any missing parents when they run, once their
dependencies are found (``_creates_directory = True``); on POSIX systems, these directories get
``jobs.permissions.directories`` when ``jobs.permissions.update`` is on. Other jobs create nothing up front, so a step
that doesn't need its directory doesn't get one; whatever their run does create there (*e.g.* a log file) counts as
theirs all the same.

When a job succeeds, it **deletes** what it created: its job directory, if the run created it, and then the parents
that creating it made, as long as they are empty (another job's files next to it keep them). A directory that existed
before the run stays, with everything in it, so a job never deletes a folder it didn't make. Nothing is deleted if the
job class sets ``_keep_files = True`` or ``jobs.keep_temp`` is on in ``xtl.toml``. When it fails, everything stays, so
its inputs and logs can be inspected. A job that writes results the user wants to keep either sets
``_keep_files = True``, or writes them outside its job directory.

Class attributes
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Default
     - Meaning
   * - ``_keep_files``
     - ``False``
     - Keep the job directory after a successful run.
   * - ``_resources``
     - ``None``
     - The job type's resource limit, used when its configuration has none (see :doc:`pools`).
   * - ``_dependencies``
     - empty
     - Names of the external programs the job needs, as in the ``[dependencies]`` section of ``xtl.toml`` (e.g.
       ``{'atsas'}``); ``dependencies()`` returns them. They are checked before the job runs, and a missing one fails
       the job with a :class:`~xtl.jobs.exceptions.DependencyError` (see :doc:`batch-jobs`).

Jobs and pools
--------------

A pool runs many jobs, up to ``max_jobs`` at a time, in one of four ways: one after the other (``SimplePool``),
concurrently in one event loop (``AsyncPool``), in worker threads (``ThreadedPool``) or in worker processes
(``MultiprocessPool``). See :doc:`pools`.

A job meets two kinds of pool:

- its **parent pool**, the pool it was submitted to, as ``self.pool`` (``None`` for a job run directly);
- **child pools**, which the job opens itself with :meth:`self.get_pool() <xtl.jobs.jobs.Job.get_pool>`, e.g. to
  run sub-tasks in parallel. A child pool's jobs share the resources of the job that opened it, and their logs
  appear under its log.

  .. code-block:: python

      async def _execute(self):
          async with self.get_pool('threads', max_jobs=4) as pool:
              pool.submit(StatsJob, configs=[StatsConfig(values=v) for v in self.config.groups])
              return [r.data for r in await pool.launch()]

The jobs of a pool can share three kinds of objects, by name, through their parent pool:

- **locks**, to let one job at a time do something: ``async with self.lock(name):`` (or the pool's default lock
  when no name is given); outside a pool, it locks nothing
- **queues**, to pass items between jobs: ``self.pool.get_queue(name)``, with ``await put(item)`` and
  ``await get()``
- **states**, dicts shared by all jobs: ``self.pool.get_state(name)``. Each read or write is atomic, but a
  read-modify-write such as ``state['n'] += 1`` needs a lock around it.

They work the same in all four pool types, so a job doesn't need to know which pool runs it:

.. code-block:: python

    async def _execute(self):
        totals = self.pool.get_state('totals')
        async with self.lock():
            totals['n'] = totals.get('n', 0) + len(self.config.values)

.. _jobs-in-workers:

Thread and process pools gotchas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ThreadedPool`` and ``MultiprocessPool`` don't run the job instances you submit: they rather serialize and submit the
job to a worker, which builds a new job instance. Hence:

- **Define job and configuration classes at module level**, in an importable module: a process worker imports them
  by name, and can't find a class defined inside a function or in a script run as ``python script.py``.
- **Put everything a job needs in its configuration.** Attributes set on the job object don't reach the worker.
- **Read the outcome from the results** that ``launch()`` returns: the submitted job object isn't updated.
- **Create every named lock, queue and state with the pool before launching** (``pool.get_state('totals')``): a
  worker can't create new ones, and raises :class:`KeyError` for a name the pool doesn't have. Only the default lock is
  always there.
- **Use picklable values in a process pool:** results, and what goes into queues and states, cross from one process
  to another.
- **Don't rely on module state** set at run time (globals, monkeypatches) in a process pool: a worker process starts
  from a fresh import on Windows and macOS. ``xtl.settings`` is the exception: the pool sends the parent's settings
  with each job.

:doc:`internals` shows what travels.
