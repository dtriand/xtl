Checklist for a new job
=======================

The rules from the other pages, to go through when adding a job to xtl or to your own package. The jobs in
``xtl.scattering.jobs`` and ``xtl.nanobragg.jobs`` are worked examples.

Classes
-------

.. rst-class:: checklist

- The configuration subclasses :class:`~xtl.jobs.config.JobConfig` (or :class:`~xtl.jobs.config.BatchJobConfig`),
  and every field is an ``Option`` with a ``desc``. A stepped job can use
  :class:`~xtl.jobs.config.SteppedJobConfig` as it is.
- The job names its configuration in brackets: ``class MyJob(Job[MyConfig])``.
- Both classes are defined at module level in an importable module (:ref:`jobs-in-workers`).
- Everything the job needs is in its configuration: no attributes set on the job from outside, no globals set at run
  time.
- Paths are ``Path`` fields, with ``path_exists=True`` for inputs.

Work and results
----------------

.. rst-class:: checklist

- ``_execute()`` returns the job's data, picklable for process pools, and raises an exception when it fails.
- The job logs through ``self.logger``, with ``%``-style arguments (``self.logger.info('Read %d frames', n)``), not
  f-strings or ``print()``.
- Long blocking calls in a job meant for ``AsyncPool`` go through :func:`asyncio.to_thread`.
- Cleanup that must happen when the job is cancelled is in a ``finally`` block.

Files
-----

.. rst-class:: checklist

- Working files go in ``config.job_directory``.
- Results the user keeps go outside it (e.g. an ``output`` field), or the job class sets ``_keep_files = True``.

Batch jobs
----------

.. rst-class:: checklist

- There is a template for the shell of every system the job runs on (``Shell.BASH``, and ``Shell.CMD`` for Windows).
- Template lines end with ``__XTL_NL__``.
- Template values come from ``get_context()``; paths are ``Path`` objects (quoted for the shell), and every other
  value is safe on a command line.
- ``dependencies`` names the programs, as in the ``[dependencies]`` section of ``xtl.toml``, so the modules site can
  load them.
- ``filename`` names the batch file after the program, so that it and its logs are easy to recognize.

Stepped jobs
------------

.. rst-class:: checklist

- Each step's options come from ``defaults`` and ``dynamic_defaults`` (reading ``ctx.parent.config`` and
  ``ctx.data``); ``post_processor`` passes on only what later steps or the caller need.
- Steps whose files may help debug a later step's failure set ``_keep_files = True``.

Pools and shared state
----------------------

.. rst-class:: checklist

- Locks, queues and states the jobs use are created on the pool before launching (the default lock always exists).
- A job that opens pools of its own has a resource request (``_resources`` or ``config.resources``).

Tests
-----

.. rst-class:: checklist

- The job runs in a ``SimplePool`` and in a ``MultiprocessPool`` (the strictest), with its records reaching a
  handler added with ``pool.add_handler()``.
- Its configuration survives ``to_dict()`` and ``from_dict()``.
- A failing run returns a result with the error, and leaves the job directory in place.
