.. |SteppedJob| replace:: :class:`SteppedJob <xtl.jobs.jobs.SteppedJob>`
.. |SteppedJobConfig| replace:: :class:`SteppedJobConfig <xtl.jobs.config.SteppedJobConfig>`
.. |StepSpec| replace:: :class:`StepSpec <xtl.jobs.steps.StepSpec>`
.. |JobContext| replace:: :class:`JobContext <xtl.jobs.steps.JobContext>`
.. |SteppedJobResults| replace:: :class:`SteppedJobResults <xtl.jobs.results.SteppedJobResults>`

Stepped jobs
============

A |SteppedJob| runs other jobs, its *steps*, one after the other. Each step is an ordinary job (a batch job, a Python
job, even another stepped job), and its configuration can be built from what the earlier steps produced.

Example
-------

A batch step finds the user's name, then the Python job from the quickstart greets them:

.. code-block:: python

    from xtl_core.options import Option
    from xtl.jobs import BatchJob, BatchJobConfig, Shell, SteppedJob, SteppedJobConfig, StepSpec


    class WhoAmIConfig(BatchJobConfig):
        templates: dict[Shell, str] = Option(
            default_factory=lambda: {
                Shell.BASH: 'echo "$USER"',
                Shell.CMD: 'echo %USERNAME%',
                Shell.POWERSHELL: 'Write-Output $env:USERNAME',
            },
            desc='Batch file content per shell',
        )


    class WhoAmIJob(BatchJob[WhoAmIConfig]):
        pass


    class Welcome(SteppedJob[SteppedJobConfig]):
        _steps = (
            StepSpec(
                name='whoami',
                job_cls=WhoAmIJob,
                config_cls=WhoAmIConfig,
                post_processor=lambda results, ctx: {'user': results.data.stdout.strip()},
            ),
            StepSpec(
                name='greet',
                job_cls=GreetJob,  # the Python job from the quickstart
                config_cls=GreetConfig,
                dynamic_defaults=lambda ctx: {'name': ctx.data['user']},
            ),
        )


    result = await Welcome(SteppedJobConfig()).run()
    result.data                 # {'user': 'alice'}
    result.steps['greet'].data  # 'Hello, alice!'

    # The caller can still set any step's options
    result = await Welcome(SteppedJobConfig(steps={'greet': {'name': 'Ada'}})).run()
    result.steps['greet'].data  # 'Hello, Ada!'

The first run returns:

.. code-block:: python

    SteppedJobResults(
        job_id='m90kC',
        data={'user': 'alice'},
        error=None,
        steps={
            'whoami': JobResults(
                job_id='m90kC.1',
                data=BatchResults(stdout='alice\n', stderr='', return_code=0),
                error=None,
            ),
            'greet': JobResults(job_id='m90kC.2', data='Hello, alice!', error=None),
        },
    )

``Welcome`` uses |SteppedJobConfig| as it is: the steps' options come from their specs, so a stepped job needs no
configuration class of its own. Subclass it only to give the stepped job options of its own, which the steps can read
as ``ctx.parent.config`` (see `The context`_).

Steps
-----

``_steps`` is a tuple of |StepSpec|, in the order they run:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``name``
     - The step's name, unique within ``_steps``: the key of its results and of its options in
       ``SteppedJobConfig.steps``.
   * - ``job_cls``, ``config_cls``
     - The step's job class and its configuration class.
   * - ``defaults``
     - Fixed options for the step.
   * - ``dynamic_defaults(ctx)``
     - A function that returns more options, from the parent's configuration or the earlier steps (see `The context`_).
   * - ``condition(ctx)``
     - Run the step only if this returns ``True`` (default: always).
   * - ``post_processor(results, ctx)``
     - Turn the step's results into a dict, which the following steps read as ``ctx.data``.

A step's configuration is built just before it runs, each layer over the one before:

#. ``defaults``;
#. ``dynamic_defaults(ctx)``;
#. the options the caller gave for that step, in the parent's ``SteppedJobConfig.steps``.

If the parent has a ``job_directory``, each step works in ``<job_directory>/<i>_<name>`` (e.g. ``1_whoami``).

The context
-----------

The functions of each step get a |JobContext|, ``ctx``:

- ``ctx.parent``: the stepped job; ``ctx.parent.config`` is its configuration;
- ``ctx.results``: the results of the steps so far, by name;
- ``ctx.data``: what the post-processors returned so far;
- ``ctx.step``: the step that just ran (in ``post_processor``).

Running
-------

The steps run one after the other, inside the stepped job's run, and **stop at the first step that fails**. Their log
records appear under the stepped job's (:doc:`logging`).

In a pool, a stepped job counts as one job: its steps run in the same thread or process as the stepped job, and
``max_jobs`` doesn't count them. To run tasks in parallel from a step, open a pool inside it (:doc:`pools`).

Results
-------

:meth:`SteppedJob.run() <xtl.jobs.jobs.SteppedJob.run>` returns a |SteppedJobResults|:

- ``steps``: each step's results, by name (steps that were skipped or never reached are missing);
- ``data``: everything the post-processors returned;
- ``error``: the error of the step that failed, if any.

If one of a step's functions raises, or the options for a step are invalid, the stepped job fails with that error.
Either way, ``steps`` and ``data`` keep what the steps before it produced.

Each step deletes its directory as soon as it succeeds, like any job (:doc:`jobs`). If a later step may need its files
to debug a failure, set ``_keep_files = True`` on the step's job class.

A stepped job's ``dependencies()`` are its own and those of its steps' job classes, and of their steps. The job checks
them all before its first step, so that a missing program fails it before any step runs. Since a step's options are
only known when the step starts, this check uses the compute site and shell that the dependencies set in ``xtl.toml``,
or the defaults; a batch step whose options pick another site or shell checks its dependencies again when it starts
(otherwise, the outcome is remembered and nothing runs twice).
