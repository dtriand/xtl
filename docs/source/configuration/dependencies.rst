Shells, compute sites and dependencies
======================================
Some jobs in XTL run external programs (*e.g.* ATSAS, autoPROC). XTL writes a batch file for each of them, in a
*shell*, and runs it on a *compute site*, which sets up the programs the job depends on. This page lists what each
shell, compute site and dependency expects to find on your machine, and how to configure them in ``xtl.toml``.

Before a job runs, XTL checks that every program it needs can be found, after the setup of the compute site; each
batch file also checks again before its first command (it stops with exit code 127 and ``xtl: <program> not found``).
Checks can be switched off with:

.. code-block:: toml

   [dependencies]
   check = false

Checking your setup
-------------------
``xtl.config deps`` checks your setup without running a job: which shells are installed, and, for each dependency (or
only those you name), whether it is found with the compute site and shell that a job needing it would use. Each check
runs afresh, also when ``check = false``. ``--compute-site`` and ``--shell`` try others:

.. code-block:: console

   $ xtl.config deps atsas autoproc
   Shells
   Shell          Executable                                                 Found
   bash           /bin/bash                                                  No
   cmd (default)  C:\Windows\System32\cmd.exe                                Yes
   powershell     C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe  Yes

   Dependencies
   Name      Compute site  Shell  Setup                Found
   atsas     modules       cmd    modules atsas/4.1.3  Yes
   autoproc  modules       cmd    PATH                 No

   Missing dependencies:
   - executables aP_wf_process, process not found (required for autoproc)
   Check dependencies in xtl.toml (see xtl.config where), or name only the dependencies you use.

The command exits with code 1 if a dependency is missing; ``-v`` shows what each check printed, *e.g.* the error of a
module that didn't load.

Shells
------
.. shells::
   :bash: The default on Linux and macOS. Batch files run without reading a profile.
   :cmd: The default on Windows.
   :powershell: Windows PowerShell 5.1. Batch files run with your PowerShell profile.

Compute sites
-------------
The compute site is set with ``jobs.compute_site``, or per dependency (see `Dependencies`_).

``local``
^^^^^^^^^
Runs jobs on this machine. Programs are found on ``PATH``, or in a directory given by ``path``: ``dependencies.path``
for every job, and each dependency's own ``path``, which come first on ``PATH`` (a dependency's before the common
one).

``modules``
^^^^^^^^^^^
Runs jobs on this machine, after loading the modules of their dependencies with
`Environment Modules <https://modules.readthedocs.io/>`_. Each batch file starts with ``module purge``, then loads ``dependencies.modules`` (for every job), then each dependency's ``modules``,
in their order. The module command must be available in the shell the job runs in:

.. tab-set::
   :sync-group: shell

   .. tab-item:: Bash
      :sync: bash

      ``module`` is a shell function, which the batch file gets from the shell that started XTL (Environment Modules
      exports it). Otherwise, XTL sets it up from ``$MODULESHOME/init/bash``, if ``MODULESHOME`` is
      set.

   .. tab-item:: PowerShell
      :sync: pwsh

      Environment Modules' command in PowerShell is ``envmodule`` (``module`` is PowerShell's own). It comes from
      Environment Modules' init script, which your PowerShell profile must run (``$PROFILE`` shows where the profile
      is). Windows PowerShell 5.1, which XTL uses, also needs ``$IsWindows`` set before it, as the init script looks
      for it to tell Windows apart:

      .. code-block:: powershell
         :caption: ``Microsoft.PowerShell_profile.ps1``

         if ($null -eq $IsWindows) { $IsWindows = $true }
         . "C:\Program Files\Environment Modules\init\pwsh.ps1"

      Without it, the check reports the modules as not loaded, with PowerShell's error for ``envmodule``.

   .. tab-item:: cmd
      :sync: cmd

      ``module`` is Environment Modules' ``module.cmd``, which sets itself up; its directory
      (``C:\Program Files\Environment Modules\bin``) must be on ``PATH``, as the installer leaves it.

Modules are found in the directories of ``MODULEPATH``, as set up by your installation (*e.g.* ``module use``, or the
``MODULEPATH`` environment variable on Windows).

Dependencies
------------
Each dependency has a table in ``xtl.toml``, with these keys:

.. dependency-keys::

The programs that XTL looks for are fixed for each dependency:

.. dependency-programs::

For example, ATSAS from a module and easyBragg from a directory:

.. code-block:: toml
   :caption: ``xtl.toml``

   [jobs]
   compute_site = "modules"

   [dependencies.atsas]
   modules = ["atsas/4.1.3"]

   [dependencies.easybragg]
   path = "/opt/easybragg/bin"
   compute_site = "local"
