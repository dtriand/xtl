.. |add_handler| replace:: :meth:`add_handler() <xtl.jobs.pools.BasePool.add_handler>`
.. |add_job_handler| replace:: :meth:`add_job_handler() <xtl.jobs.pools.BasePool.add_job_handler>`

Logging
=======

Jobs log with the standard :mod:`logging` module, through ``self.logger``. To see their records, attach handlers to the
pool, which then get the records of every job, or to each job, for example a log file in its directory.

The logger tree
---------------

Each pool's logger is the root of a tree:

.. code-block:: text

    ThreadedPool#Xa3f9                        the pool (<Class>#<pool id>, or <Class>|<name>#<pool id> if named)
    ├── FitJob|1                              a job (its label; its id is FitJob-1)
    │   ├── FitJob|1.1                        its steps
    │   └── FitJob|1.2
    └── SweepJob|1
        └── ThreadedPool#Qk27                 a pool opened inside the job
            └── PointJob|1                    ... and its jobs

The full logger names join these with dots (``ThreadedPool#Xa3f9.FitJob|1.2``). Records travel up the tree to the
pool's handlers, also from worker threads and processes, and stop there: they never reach the application's root
logger. A job run without a pool has a logger of its own, named after its label (``FitJob|1``).

Handlers for the pool
---------------------

A handler added to a pool gets the records of the pool, of all its jobs and steps, and of the pools opened inside
them:

.. code-block:: python

    import logging

    from xtl.jobs import AsyncPool

    handler = logging.FileHandler('run.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(job)s] %(message)s'))

    async with AsyncPool(max_jobs=4) as pool:
        pool.add_handler(handler)
        pool.submit(StatsJob, configs=[StatsConfig(values=[1, 2, 3]), StatsConfig(values=[2, 4, 6])])
        await pool.launch()

``run.log`` then reads:

.. code-block:: text

    2026-11-17 10:00:00,001 INFO [StatsJob|1] 3 values: mean 2.00, standard deviation 0.82
    2026-11-17 10:00:00,002 INFO [StatsJob|2] 3 values: mean 4.00, standard deviation 1.63

``%(job)s`` is the logger's name within the pool. For the tree above:

.. list-table::
   :header-rows: 1

   * - Record from
     - ``%(job)s``
     - Full logger name (``%(name)s``)
   * - the pool itself
     - ``ThreadedPool`` (``ThreadedPool|refine`` if named)
     - ``ThreadedPool#Xa3f9``
   * - a job
     - ``FitJob|1``
     - ``ThreadedPool#Xa3f9.FitJob|1``
   * - a step
     - ``FitJob|1.2``
     - ``ThreadedPool#Xa3f9.FitJob|1.2``
   * - a job in a pool opened by a job
     - ``SweepJob|1>PointJob|1``
     - ``ThreadedPool#Xa3f9.SweepJob|1.ThreadedPool#Qk27.PointJob|1``
   * - a logger outside the pool
     - its full name
     - its full name

The handler is yours: the pool detaches it on exit, but doesn't close it.

A pool also prints INFO and above to ``stdout``. To use only your handlers instead, give them to the pool in a
:class:`~xtl.jobs.logging.LoggerConfig`:

.. code-block:: python

    from xtl.jobs.logging import LoggerConfig

    async with AsyncPool(max_jobs=4, logger_config=LoggerConfig(handlers=[handler])) as pool:
        ...

Handlers for each job
---------------------

|add_job_handler| gives every job its own handler, built from a configuration. The most common is a log file in the
job's directory:

.. code-block:: python

    from xtl.jobs.logging import JobFileHandlerConfig

    pool.add_job_handler(JobFileHandlerConfig(), name='file')  # <job_directory>/xtl.log, from DEBUG up

Any handler configuration works, with placeholders that each job fills in: ``{job_directory}``, ``{job_id}``,
``{job_class}`` and ``{pool}``:

.. code-block:: python

    from xtl.jobs.logging import FileHandlerConfig

    pool.add_job_handler(FileHandlerConfig(filename='logs/{job_id}.log', level=logging.DEBUG), name='per-job')
    # logs/StatsJob-1.log, logs/StatsJob-2.log, ... (the logs directory must exist)

A job that has no value for a placeholder (e.g. no ``job_directory``) simply gets no such handler; one whose handler
fails to start (e.g. a missing directory) logs a warning and runs without it. A job's own handler
gets the records of its steps too. Each handler has its own level, so a DEBUG log file doesn't make the terminal show
DEBUG.

A job's configuration can turn one of these off, or replace it, by its name:

.. code-block:: python

    StatsConfig(values=[1, 2, 3], log_handlers={'file': None})  # no xtl.log for this job

For a handler class of your own, pass it as ``handler`` with its arguments in ``options``. The class must be importable
(defined at module level), since the configuration travels to worker threads and processes:

.. code-block:: python

    from xtl.jobs.logging import HandlerConfig

    pool.add_job_handler(HandlerConfig(handler=MyHandler, options={'path': '{job_directory}/events.json'}), name='events')

In a process pool, a handler in the main process gets a worker's exceptions as text (``record.exc_text``), not as
exception objects.

Settings
--------

- ``jobs.tracebacks``: log the traceback of a failing job.
- ``cli.job_log_files``: whether the CLIs give each job an ``xtl.log``.
