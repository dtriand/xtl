.. |BatchJob| replace:: :class:`BatchJob <xtl.jobs.jobs.BatchJob>`
.. |BatchJobConfig| replace:: :class:`BatchJobConfig <xtl.jobs.config.BatchJobConfig>`
.. |BatchResults| replace:: :class:`BatchResults <xtl.jobs.results.BatchResults>`
.. |Shell| replace:: :class:`Shell <xtl.jobs.shells.Shell>`
.. |ComputeSite| replace:: :class:`ComputeSite <xtl.jobs.sites.ComputeSite>`
.. |SubprocessError| replace:: :class:`SubprocessError <xtl.jobs.exceptions.SubprocessError>`

Batch jobs
==========

A |BatchJob| runs an external program. It renders a script, the *batch file*, from a template, saves it in its job
directory, and runs it, with the program's output going to two log files. The batch file stays on disk, so every run
can be repeated by hand.

A batch job needs no code of its own, only a configuration with templates.

Example
-------

A job that greets the user, with one template per shell:

.. code-block:: python

    from xtl_core.options import Option
    from xtl.jobs import BatchJob, BatchJobConfig, Shell


    class GreetConfig(BatchJobConfig):
        greeting: str = Option(default='Hello', desc='How to greet')
        templates: dict[Shell, str] = Option(
            default_factory=lambda: {
                Shell.BASH: 'echo "__GREETING__, $USER"',
                Shell.CMD: 'echo __GREETING__, %USERNAME%',
                Shell.POWERSHELL: 'Write-Output "__GREETING__, $env:USERNAME"',
            },
            desc='Batch file content per shell',
        )

        def get_context(self) -> dict:
            return super().get_context() | {'GREETING': self.greeting}


    class GreetJob(BatchJob[GreetConfig]):
        pass


    result = await GreetJob(GreetConfig(greeting='Hi')).run()
    result.data.stdout       # 'Hi, alice\n'
    result.data.return_code  # 0

On the local compute site, the job writes one of these batch files, depending on the shell:

.. tab-set::
   :sync-group: shell

   .. tab-item:: Bash
      :sync: bash

      ``batch_job.sh``

      .. code-block:: bash

          #!/bin/bash

          echo "Hi, $USER"

   .. tab-item:: PowerShell
      :sync: pwsh

      ``batch_job.ps1``

      .. code-block:: powershell

          Write-Output "Hi, $env:USERNAME"

   .. tab-item:: cmd
      :sync: cmd

      ``batch_job.bat``

      .. code-block:: batch

          chcp 65001 > nul

          echo Hi, %USERNAME%

``result.data`` is a |BatchResults|: the program's ``stdout`` and ``stderr`` as strings, and its ``return_code``. A
non-zero return code makes the job fail with a |SubprocessError|.

Configuration
-------------

|BatchJobConfig| adds these fields to those of every job:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Field
     - Default
     - Meaning
   * - ``templates``
     - empty
     - The batch file's content, one template per |Shell| (see `Templates`_).
   * - ``arguments``
     - empty
     - Arguments of the batch file on its command line (``$1``, ``%1``), each passed as one argument.
   * - ``default_shell``
     - ``None``
     - The shell to use, if the templates offer several (see `Shells`_).
   * - ``compute_site``
     - ``jobs.compute_site`` in ``xtl.toml``
     - Where and how the batch file runs (see `Compute sites`_).
   * - ``dependencies``
     - empty
     - Names of the programs the job needs, as in the ``[dependencies]`` section of ``xtl.toml`` (e.g.
       ``{'atsas'}``). The modules site loads their modules.
   * - ``job_directory``
     - a new temporary directory, ``xtl_batch_*``
     - Where the batch file and its logs go. Deleted after a successful run, like any job directory
       (:doc:`jobs`).
   * - ``filename``
     - ``'batch_job'``
     - Name of the batch file and its logs, without extension.
   * - ``permissions``
     - ``jobs.permissions.scripts`` in ``xtl.toml``
     - Permissions of the batch file; the owner must be able to execute it.

It also has these properties:

- ``shell``: the shell the batch file is written for (see `Shells`_);
- ``file``: the batch file, ``<job_directory>/<filename><extension>``, e.g. ``batch_job.sh``;
- ``stdout`` and ``stderr``: the log files, ``<job_directory>/<filename>.stdout.log`` and ``.stderr.log``.

Templates
---------

A template is the body of the batch file, written in the shell's own syntax, with placeholders that look like
``__NAME__``. This one lists the data files in a directory and checks that an interpreter exists:

.. code-block:: python

    from pathlib import Path
    import sys

    from xtl_core.options import Option
    from xtl.jobs import BatchJobConfig, Shell


    class ListConfig(BatchJobConfig):
        data_dir: Path = Option(default=Path('/data/my run'), desc='Where the data files are')
        pattern: str = Option(default='*.dat', desc='Which files to list')
        templates: dict[Shell, str] = Option(
            default_factory=lambda: {
                Shell.BASH: '__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__'
                            'cd __DATA_DIR____XTL_NL__'
                            'for f in __PATTERN__; do echo "$f"; done__XTL_NL__'
                            'command -v __PYTHON__',
            },
            desc='Batch file content per shell',
        )

        def get_context(self) -> dict:
            return super().get_context() | {
                'DATA_DIR': self.data_dir,       # a Path: quoted accordingly
                'PATTERN': self.pattern,         # a str: as it is, so that the shell expands it
                'PYTHON': Path(sys.executable),  # the same for every job
            }

The batch file:

.. code-block:: bash

    #!/bin/bash

    # Generated by xtl 0.2.0 on 2026-11-17T10:00:00.000000
    cd '/data/my run'
    for f in *.dat; do echo "$f"; done
    command -v /path/to/xtl/python

End each line with ``__XTL_NL__``, the shell's line ending, rather than ``\n``. For a literal ``__``, write ``____``.

Values come from two places:

- the configuration's :meth:`~xtl.jobs.config.BatchJobConfig.get_context`, a dict of placeholder names to values. A
  configuration adds its own values to its parent's (``super().get_context() | {...}``). Values that don't depend on
  the configuration's fields, like ``PYTHON`` above, go here too, so that they reach jobs in worker threads and
  processes, which only receive the configuration.
- three placeholders that are always there: ``__XTL_COMMENT__`` (the shell's comment, ``#`` or ``rem``),
  ``__XTL_NL__`` (the shell's line ending) and ``__XTL_DOCSTRING__`` (``Generated by xtl <version> on <time>``).

A placeholder without a value raises :class:`KeyError` when the job runs.

.. important::

   Values are inserted **as they are**, except :class:`~pathlib.Path` objects, which are quoted for the shell (above,
   ``DATA_DIR`` has a space). Pass paths as ``Path``, and make sure that other values are safe on a command line:
   spaces, quotes or ``;`` in a value from user input can break the command, or run another one.

**The keys of** ``templates`` **are the shells the job supports.** A job runs only with one of them: if the shell
picked for it has no template, creating the configuration raises :class:`RuntimeError`. A job with only a ``BASH``
template, for example, can't be configured on Windows unless ``default_shell`` is ``Shell.BASH``.

Shells
------

.. list-table::
   :header-rows: 1

   * - |Shell|
     - Batch file
     - Runs it as
   * - ``BASH``
     - ``.sh``
     - ``/bin/bash <file> <arguments>``
   * - ``CMD``
     - ``.bat``
     - ``cmd.exe /Q /C call <file> <arguments>``
   * - ``POWERSHELL``
     - ``.ps1``
     - ``powershell.exe -File <file> <arguments>``

The configuration picks the shell when it's created: ``default_shell`` if it's set, otherwise the system's shell,
``CMD`` on Windows and ``BASH`` elsewhere. The built-in compute sites accept every shell; a site that accepts only some
picks one of those that the templates have, ``default_shell`` first.

Batch files are written in UTF-8. cmd reads a batch file in the console's code page, so a cmd batch file starts with
``chcp 65001 > nul``, which switches the console to UTF-8 (for paths with ``é`` or ``Δ``, for instance). The console
keeps that code page after the batch file ends.

Running a batch job
-------------------

A batch job runs like any other job (:doc:`jobs`), alone or in a pool (:doc:`pools`). It needs no ``_execute()``: the
|BatchJob| class renders the template, saves the batch file in ``job_directory``, runs it on the compute site and reads
the two log files into a |BatchResults|.

The batch file runs:

- in the working directory of the Python process, not the job directory, so templates should use absolute paths
  (pass them as ``Path``);
- without input: a program that reads its input gets end-of-file, instead of waiting for the keyboard;
- with its output copied to the log files as it comes. The log files hold the program's raw output: they stay out of
  the job's log, and are kept or deleted with the job directory.

If the job is cancelled, or Ctrl+C is pressed, the batch file and every program it started are stopped, and the job
ends cancelled. On POSIX, the programs get ``SIGINT``, then ``SIGTERM`` and ``SIGKILL`` 5 seconds apart if they still
run. On Windows, they get ``Ctrl+Break``, and after 5 seconds whatever is left is ended (``taskkill``).

Compute sites
-------------

A compute site decides what goes around the template in the batch file, and how it runs. Set the default with
``jobs.compute_site`` in ``xtl.toml``, or pick one per job with ``compute_site=ComputeSite.LOCAL`` in the
configuration.

A batch file has three parts, each written by the compute site:

- the **preamble**: what the shell and the site need before anything else, such as a shebang, the setup of the job's
  dependencies (directories put on ``PATH``, ``module load`` lines), and a guard line for each program they provide,
  which ends the batch file with exit code 127 and ``xtl: <program> not found`` if the program isn't found, before
  any command runs;
- the **content**: the rendered template;
- the **postamble**: what comes after it; empty on the built-in sites.

Local
^^^^^

``ComputeSite.LOCAL`` runs the batch file on this machine. The programs must be on ``PATH``, or in a directory given
by ``path`` in ``xtl.toml``: ``dependencies.path`` for every job, and each dependency's own (e.g.
``dependencies.atsas.path``), which the preamble puts first on ``PATH``, a dependency's before the common one. Without
dependencies, the preamble is only the shell's first line: ``#!/bin/bash`` for bash, ``chcp 65001 > nul`` for cmd,
nothing for PowerShell. The batch files in the `Example`_ are from this site.

Modules
^^^^^^^

``ComputeSite.MODULES`` runs the batch file on this machine too, but first loads the job's programs with
`Environment Modules <https://modules.readthedocs.io/>`_. The modules come
from ``xtl.toml``: those in ``dependencies.modules`` are loaded for every job, then those of each dependency in the
configuration's ``dependencies``, each in its order; directories in ``path`` are put on ``PATH`` after that. In
PowerShell, Environment Modules' command is ``envmodule``, which comes from the PowerShell profile (batch files run
with it); bash sets up ``module`` from ``$MODULESHOME`` if the shell that started the job didn't pass it on. With this
``xtl.toml``:

.. code-block:: toml

    [dependencies]
    modules = ['gcc/12.2.0']

    [dependencies.atsas]
    modules = ['atsas/3.2.1']

the example job with ``dependencies={'atsas'}`` writes:

.. tab-set::
   :sync-group: shell

   .. tab-item:: Bash
      :sync: bash

      .. code-block:: bash

          #!/bin/bash
          type module > /dev/null 2>&1 || { [ -f "$MODULESHOME/init/bash" ] && . "$MODULESHOME/init/bash"; }
          module purge
          module load gcc/12.2.0
          module load atsas/3.2.1
          command -v datcmp > /dev/null 2>&1 || { echo 'xtl: datcmp not found' >&2; exit 127; }
          command -v datop > /dev/null 2>&1 || { echo 'xtl: datop not found' >&2; exit 127; }

          echo "Hi, $USER"

   .. tab-item:: PowerShell
      :sync: pwsh

      .. code-block:: powershell

          envmodule purge
          envmodule load gcc/12.2.0
          envmodule load atsas/3.2.1
          if (-not (Get-Command datcmp -ErrorAction SilentlyContinue)) { [Console]::Error.WriteLine('xtl: datcmp not found'); exit 127 }
          if (-not (Get-Command datop -ErrorAction SilentlyContinue)) { [Console]::Error.WriteLine('xtl: datop not found'); exit 127 }

          Write-Output "Hi, $env:USERNAME"

   .. tab-item:: cmd
      :sync: cmd

      .. code-block:: batch

          chcp 65001 > nul
          call module purge
          call module load gcc/12.2.0
          call module load atsas/3.2.1
          where datcmp > nul 2>&1 || (echo xtl: datcmp not found 1>&2 & exit /b 127)
          where datop > nul 2>&1 || (echo xtl: datop not found 1>&2 & exit /b 127)

          echo Hi, %USERNAME%

The lines before the blank line are the preamble (the last two, the guards for ATSAS's ``datcmp`` and ``datop``), the
last one the content.

Checking dependencies
^^^^^^^^^^^^^^^^^^^^^

A compute site also checks that dependencies can be found, before a job runs:
:meth:`~xtl.jobs.sites.LocalSite.check_dependencies` runs, for each dependency on its own, a short script in the job's
shell with the same setup as the preamble, loading each module and then looking up every program the dependency
provides. It returns a :class:`~xtl.jobs.dependencies.DependencyCheck` per dependency, which says what's missing:
modules that didn't load, programs that weren't found. The outcome is remembered in the dependency's settings, for that
site and shell, until the dependency's settings change (``settings.dependencies.clear_checks()`` forgets them all).

.. code-block:: python

    from xtl.jobs.shells import Shell
    from xtl.jobs.sites import ModulesSite

    for check in ModulesSite().check_dependencies(['atsas'], shell=Shell.CMD.get()):
        print(check.dependency, check.found, check.missing)

Sites that submit jobs to a scheduler check on the machine that submits them; the guard lines check again on the node
that runs the job.

Command policies
----------------

A command policy (:mod:`xtl.jobs.policies`) changes a batch file's parts when the batch file is saved, after the
compute site has written them: the preamble, each command of the content (blank lines aren't commands), the whole
content, and the postamble. It's a way to apply a rule to every batch file of a site, e.g. to run every command with
lower priority (:class:`~xtl.jobs.policies.NiceCommandPolicy` puts ``nice -n 10`` before each command).

A policy implements the four ``intercept_*`` methods, each returning its part, changed or not. This one makes a bash
batch file stop at the first failing command, and say so if none failed:

.. code-block:: python

    from xtl.jobs.policies import BaseCommandPolicy
    from xtl.jobs.sites import LocalSite


    class StopOnErrorPolicy(BaseCommandPolicy):
        name = 'stop_on_error'

        def intercept_preamble(self, preamble: str) -> str:
            return preamble + 'set -e\n'

        def intercept_command(self, command: str) -> str:
            return command

        def intercept_content(self, content: str) -> str:
            return content

        def intercept_postamble(self, postamble: str) -> str:
            return postamble + 'echo "All commands succeeded"'


    class StrictSite(LocalSite):
        _policy = StopOnErrorPolicy()

A batch file with the commands ``cd /data`` and ``process`` then becomes:

.. code-block:: bash

    #!/bin/bash
    set -e

    cd /data
    process
    echo "All commands succeeded"

A site gets its policy from its ``_policy`` class attribute; the built-in sites have none.

.. note::

   A configuration's ``compute_site`` accepts only the built-in sites, so a policy can't be applied to a batch job
   yet.
