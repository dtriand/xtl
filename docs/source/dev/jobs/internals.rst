.. |JobSubmission| replace:: :class:`JobSubmission <xtl.jobs.submissions.JobSubmission>`
.. |JobData| replace:: :class:`JobData <xtl.jobs.jobs.JobData>`

Under the hood
==============

This page follows a job from ``submit()`` to its result, in each pool type. It is for changing :mod:`xtl.jobs`
itself, or for debugging a job that behaves differently in one pool type than in another.

What travels to a worker
------------------------

:meth:`~xtl.jobs.pools.BasePool.submit` creates, for each configuration, the job object (with its logger, a child of
the pool's) and a |JobSubmission|: everything needed to run the job somewhere else.

- ``data``: a |JobData| with the job class as an import path, the job id, the configuration as a dict, the pool's
  job handler specs and the pool's label
- ``ipc``: the locks, queues and states registered with the pool when the job starts
- ``resources``: what the job may use (its request, capped by the pool's share)
- ``logger_name`` and ``log_level``: its logger's full name and level
- ``xtl_settings`` (process pools only): a snapshot of ``xtl.settings``, taken when the job starts

Nothing else about the job object travels: no attributes set on it, no logger configuration. A worker rebuilds the
job with :meth:`Job.deserialize() <xtl.jobs.jobs.Job.deserialize>`, which imports the class and validates the
configuration again.

Running a job in each pool type
-------------------------------

:meth:`~xtl.jobs.pools.BasePool.launch` creates one :mod:`asyncio` task per submission. Each task waits for a slot (a
semaphore of ``max_jobs``), runs the submission, and sets the job's status as it goes. In thread and process pools,
the task awaits the worker, so the status is always set in the main process.

``SimplePool`` and ``AsyncPool``
   Run the submitted job object itself, in the pool's event loop. ``SimplePool`` is an ``AsyncPool`` with one slot.

``ThreadedPool``
   In a worker thread: a new event loop (``asyncio.run()``), a resource manager with the submission's resources, a
   new job built from the submission (using the same logger object, since threads share loggers), a ``ProxyPool`` over
   the pool's own locks, queues and states, then ``job.run()``. The result comes back as it is.

``MultiprocessPool``
   On entry, the pool starts a ``multiprocessing`` manager process for its shared state, a log queue with a listener,
   and the ``ProcessPoolExecutor``. Each worker process resets the loggers it inherited and sends its records to the
   log queue (see `Logs from worker processes`_). For each job, the submission is pickled to a worker, which:

   #. applies the settings snapshot to its ``xtl.settings``
   #. creates a resource manager with the submission's resources
   #. builds the job (importing its class), with a ``ProxyPool`` over the manager's proxies
   #. runs ``job.run()`` in a new event loop

   The result is pickled back; one that can't be pickled is replaced by a
   :class:`~xtl.jobs.exceptions.WorkerError`.

Logs from worker processes
--------------------------

Threads share the main process's loggers, so nothing is needed for them. A worker process sends its records back:

#. When the worker starts, it resets every logger it inherited, and puts one :class:`~xtl.jobs.logging.RelayHandler`
   on its root logger.
#. A job's logger in the worker has the same full name as in the main process, and the pool's level, so records the
   pool wouldn't show stay in the worker. The job's own handlers (e.g. its ``xtl.log``) are built in the worker and
   write there directly.
#. The relay handler puts each record on the log queue as a JSON-safe dict, with the message already formatted and
   the traceback as text.
#. In the main process, the listener re-emits each record through the logger of the same name, and the tree takes it
   to the pool's handlers.
#. A job's result is returned only once all its records have been re-emitted, so its last lines come before its
   result.

So a handler in the main process sees a worker's exceptions as text (``record.exc_text``), not as exception objects.

Cancellation
------------

In the event loop, cancelling a job's task cancels the job. A job in a worker thread or process can't be reached that
way, so:

- the pool marks the job as cancelled, with the reason, in a shared state of its own (``_xtl_cancel``);
- in the worker, ``_run_watched()`` runs the job next to a watcher, which looks for the mark every 0.25 s and cancels
  the job there;
- if workers are still busy after ``_CANCEL_GRACE`` seconds (15), a process pool ends them (``_end_workers()``:
  ``terminate()``, then ``kill()``); a thread pool warns and waits.

``Ctrl+C`` reaches every process of the console or terminal. A process pool's workers and its manager ignore it from
the moment they start (``shielded_context()``), so only the main process acts on it, by cancelling the jobs as above.
Batch files run in a session (POSIX) or process group (Windows) of their own, which ``Ctrl+C`` doesn't reach either
(:doc:`batch-jobs`).

Shared state across processes
-----------------------------

A ``MultiprocessPool``'s locks, queues and states live in the manager process; jobs hold proxies to them, and every
operation is a call to the manager. Workers can't create new ones (nor can a thread pool's), hence the rule to create
them before launching.

Waiting for a lock or a queue item in a thread or process pool doesn't block the event loop: the job tries, then
sleeps and tries again, from 1 ms up to 50 ms between tries (:func:`~xtl.jobs.ipc.base.poll`), so a cancelled wait
stops cleanly. The cost is up to 50 ms of delay for a job that has waited long. The locks are not reentrant: a job
that takes a lock it already holds waits forever.

When a pool exits
-----------------

On exit, a pool cancels its jobs if the block raised, shuts down its executor, stops its log listener and shared
state, gives back its resources, and removes its logger and every logger under it from :mod:`logging`. Without that
last step, a long session would keep thousands of loggers, and every :meth:`logging.Logger.setLevel` call gets slower
with each one. Handlers that the pool built from its configuration are closed; those added with ``add_handler()`` are
only detached.

Workers started by fork
-----------------------

On Linux with Python before 3.14, a process pool's workers start as copies of the main process (*fork*), with its
loggers, context variables and anything patched at run time. Windows and macOS always start a fresh interpreter, and
Linux with Python 3.14 a fresh one from a server process. The worker setup undoes what matters for jobs (loggers, the
resource lease in scope). A bug that only shows on Linux with an older Python is probably state inherited through
fork; ``TestForkedWorker`` in ``tests/jobs/test_logging.py`` recreates that state on every system.

Extension points
----------------

- **A log handler for each job**: any importable :class:`logging.Handler` subclass, in a
  :class:`~xtl.jobs.logging.HandlerConfig` (:doc:`logging`).
- **A pool type**: subclass :class:`~xtl.jobs.pools.BasePool` and define ``_ipc_cls`` (an
  :class:`~xtl.jobs.ipc.base.IPCBackend`), ``_resolve_resources()`` (what to request for a given ``max_jobs``) and
  ``_execute_submission()`` (run one submission, return its :class:`~xtl.jobs.results.JobResults`).
- **A shell, compute site or command policy**: subclass :class:`~xtl.jobs.shells.BaseShell`,
  :class:`~xtl.jobs.sites.LocalSite` or :class:`~xtl.jobs.policies.BaseCommandPolicy`. A configuration only accepts
  the built-in shells and sites for now, so these work with a :class:`~xtl.jobs.batchfiles.BatchFile` built by hand,
  not yet in a batch job.
