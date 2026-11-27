.. |Job| replace:: :class:`Job <xtl.jobs.jobs.Job>`
.. |BatchJob| replace:: :class:`BatchJob <xtl.jobs.jobs.BatchJob>`
.. |SteppedJob| replace:: :class:`SteppedJob <xtl.jobs.jobs.SteppedJob>`
.. |JobConfig| replace:: :class:`JobConfig <xtl.jobs.config.JobConfig>`
.. |JobResults| replace:: :class:`JobResults <xtl.jobs.results.JobResults>`
.. |AsyncPool| replace:: :class:`AsyncPool <xtl.jobs.pools.AsyncPool>`

Jobs and pools
==============

:mod:`xtl.jobs` is a framework for harmonizing the execution of units of work (*jobs* hereafter). It abstracts away the
details of running jobs in threads or processes, managing inter-process communication and resources allocation. Jobs
can execute Python code or external programs and can be chained together to form pipelines.

Quickstart
----------

Each job requires two parts:

* A |Job| subclass that implements the job's logic in an ``_execute()`` coroutine, and
* A |JobConfig| subclass that carries any parametrization required for the job.

.. code-block:: python

    import asyncio

    from xtl_core.options import Option
    from xtl.jobs import AsyncPool, Job, JobConfig


    class GreetConfig(JobConfig):
        name: str = Option(default='world', desc='Who to greet')


    class GreetJob(Job[GreetConfig]):
        async def _execute(self):
            self.logger.info('Hello, %s!', self.config.name)
            return f'Hello, {self.config.name}!'


    async def main():
        # One job on its own
        job = GreetJob(GreetConfig(name='xtl'))
        result = await job.run()
        print(result.success, result.data)  # True Hello, xtl!

        # Many jobs, up to 4 at a time
        async with AsyncPool(max_jobs=4) as pool:
            pool.submit(GreetJob, configs=[GreetConfig(name=n) for n in ('Ada', 'Grace', 'Linus')])
            results = await pool.launch()
        print([r.data for r in results])  # ['Hello, Ada!', 'Hello, Grace!', 'Hello, Linus!']


    asyncio.run(main())

:meth:`Job.run() <xtl.jobs.jobs.Job.run>` never raises for a failing job: it returns a |JobResults| with the job's
return value in ``data`` or the exception in ``error``, and ``success`` tells them apart.

The pieces
----------

- **Jobs** (:doc:`jobs`): a |Job| does one piece of work in its ``_execute()``, configured by a |JobConfig|, and
  returns a |JobResults| with its data or its error.
- **Batch jobs** (:doc:`batch-jobs`): a |BatchJob| runs an external program. It writes a script from a template, for
  the right *shell* (bash, cmd, PowerShell), with what the *compute site* needs around it (e.g. ``module load``).
- **Stepped jobs** (:doc:`stepped-jobs`): a |SteppedJob| runs other jobs one after the other, each configured from
  what the earlier ones produced.
- **Pools** (:doc:`pools`): run many jobs at a time, in the event loop, in threads or in processes, sharing the
  machine's resources, and locks, queues and states between the jobs.
- **Logging** (:doc:`logging`): every pool has a tree of loggers; a handler on the pool sees the records of all its
  jobs, and each job can also write its own log file.
- **Internals** (:doc:`internals`): what a pool sends to a worker thread or process, and how it runs there.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   jobs.rst
   batch-jobs.rst
   stepped-jobs.rst
   pools.rst
   logging.rst
   internals.rst
   checklist.rst
