``xtl.autoproc``
================
The ``xtl.autoproc`` command is a wrapper around `autoPROC <https://www.globalphasing.com/autoproc/>`_ from Global
Phasing Ltd. This command helps in processing several datasets with autoPROC in batch mode, either with the same or
different parameters. In addition, it can also take care of loading the dependencies, setting up file permissions and
parsing the results, with minimum user intervention.

Prerequisites
-------------
.. caution::

    ``xtl.autoproc`` is currently only tested and officially supported on Linux systems.

In order for ``xtl.autoproc`` to run, a working installation of autoPROC is required (instructions can be found
`here <https://www.globalphasing.com/autoproc/manual/installation/>`_). This implies that all autoPROC dependencies are
also installed.

Once installed, the autoPROC ``process`` command should be discoverable in the system's PATH. You can check this by
running the following command in your terminal:

.. code-block:: console

    $ which process

If the above command returns a path, then you are good to go.

.. hint::
    If autoPROC is not available on the system PATH, then ``xtl.autoproc`` can also leverage modules (*e.g.*
    `Environment Modules <https://github.com/envmodules/modules>`_) to load the required dependencies. Detailed
    documentation will be provided in the future.

``process`` command
-------------------
The ``process`` command is the main entry point for running autoPROC on multiple datasets. It takes as input a CSV file
containing paths to the first image of each dataset. In the simplest case, the command can be run as follows:

.. code-block:: bash

    $ xtl.autoproc process datasets.csv

where ``datasets.csv`` is a CSV file containing the paths the first image of each dataset to be processed, *i.e.*

.. code-block:: csv
   :caption: ``datasets.csv``

    # fist_image
    /path/to/dataset1/dataset1_00001.cbf.gz
    /path/to/dataset2/dataset2_00001.cbf.gz
    /path/to/dataset3/dataset3_00001.cbf.gz
    ...

You will be prompted to confirm that the input parameters are correct.

.. important::

    Before proceeding **take a moment to check the output paths**. autoPROC jobs can generate hundreds of files and
    directories that might be difficult or annoying to clean up if they are at the wrong location.

A few examples
~~~~~~~~~~~~~~
.. hint::
   To print a detail list of all available options run:

   .. code-block:: console

      $ xtl.autoproc process --help

When options are provided directly to the ``process`` command, they will be applied to all datasets by default, unless
explicitly overridden in the CSV file. Some frequently used options are:

Unit-cell & space group
^^^^^^^^^^^^^^^^^^^^^^^
Specify starting unit-cell parameters and space group for indexing:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --unit-cell="78 78 37 90 90 90" --space-group="P43212"

Reference MTZ file
^^^^^^^^^^^^^^^^^^
Specify a reference MTZ file to use for unit-cell parameters, space group and R-free flags:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --mtz-ref="/path/to/reference.mtz"

Resolution cutoff
^^^^^^^^^^^^^^^^^
Apply a resolution range cutoff:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --resolution=80-1.2
  $ xtl.autoproc process datasets.csv --resolution=80-
  $ xtl.autoproc process datasets.csv --resolution=1.2

In the first case both a low and high resolution cutoff are applied, while in the other two only a low or high
resolution cutoff is applied, respectively.

Anomalous signal
^^^^^^^^^^^^^^^^
By default, ``xtl.autoproc`` will run autoPROC ``process`` with the ``-ANO`` flag, meaning that the Friedel pairs will
be kept separate. To enforce merging of Friedel pairs:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --no-anomalous

Ice rings
^^^^^^^^^
autoPROC can automatically detect and exclude ice rings from the data. In case where the datasets are heavily
contaminated with ice, one can force the ice ring exclusion:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --exlude-ice

This option will set the following two autoPROC parameters to ``True``:

.. code-block::

  XdsExcludeIceRingsAutomatically=yes
  RunIdxrefExcludeIceRingShells=yes

Beamline macros
^^^^^^^^^^^^^^^
Certain beamline-specific macros exist in autoPROC. These can be selected as follows:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --beamline="PetraIIIP14"

This is equivalent to ``-M PetraIIIP14`` when running directly the autoPROC ``process`` command.

Note that the only beamline macro files can be passed with this mechanism and not any arbitrary macro. The list of
supported beamline macros can be found *via* ``xtl.autoproc process --help``.

Dataset merging
^^^^^^^^^^^^^^^
In case of incomplete data, or multiple sweeps of the same crystal, one can process multiple datasets in a single
autoPROC run and try to merge them, if they are compatible. In ``xtl.autoproc`` this can only be achieved by providing
a ``group_id`` column in the CSV file, *e.g.*:

.. code-block:: csv
  :caption: ``datasets.csv``

  # group_id,first_image
  1,/path/to/dataset1/dataset1_00001.cbf.gz
  2,/path/to/dataset2/dataset2_00001.cbf.gz
  2,/path/to/dataset3/dataset3_00002.cbf.gz

In the above example, ``dataset2_00001.cbf.gz`` and ``dataset3_00002.cbf.gz`` will be merged into the same dataset,
since they have they same ``group_id``. This option essentially passes multiple ``-Id`` flags to the autoPROC
``process`` command (see `Multi-sweep dataset processing
<http://www.globalphasing.com/autoproc/manual/autoPROC7.html#step5>`_).

.. seealso::

  See `CSV specification`_ for more details on how to structure the input CSV file.

Output directory
^^^^^^^^^^^^^^^^
By default, the output of ``xtl.autoproc process`` will be saved in the current working directory. To specify a
different output directory:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --out-dir="/path/to/output"

This will create subdirectories for each dataset in the specified output directory.

.. note::

    ``xtl.autoproc`` internally splits the input path to the first image into a ``raw_data_dir``, ``dataset_dir`` and
    ``dataset_name`` components, so that ``first_image = {raw_data_dir}/{dataset_dir}/{dataset_name}_00001.ext``.
    Therefore, the final output directory for each dataset will be ``{output_dir}/{dataset_name}``. To further
    understand the dataset discovery process, its gotchas and how to overcome them, read the `Dataset discovery`_
    section.

A very common convention for diffraction data is to store the raw images and processed data in separate locations that
have a similar filestructure, *e.g.*:

.. code-block:: text

   RAW_DATA:  /path/to/raw_data/datasets/dataset1/dataset1_00001.cbf.gz
   PROCESSED: /path/to/processed/datasets/dataset1/

Here, the top level directory is different (``/path/to/raw_data`` vs. ``/path/to/processed``), but the file tree is the
same after some point (``datasets/dataset1/``). In this case, one can specify both ``--raw-dir`` and ``--out-dir`` to
influence the dataset discovery process and ensure that all output files will be saved in the correct subdirectory:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --raw-dir="/path/to/raw_data" --out-dir="/path/to/processed"

If we run the above command with the following CSV file:

.. code-block:: csv
   :caption: ``datasets.csv``

   # first_image
   /path/to/raw_data/datasets/dataset1/dataset1_00001.cbf.gz
   /path/to/raw_data/datasets/dataset2/dataset2_00001.cbf.gz
   /path/to/raw_data/datasets/dataset3/dataset3_00001.cbf.gz

then the output directories will be ``/path/to/processed/datasets/datasetX``.

.. seealso::

    See the `Dataset discovery`_ section for more details.

Parallelization
^^^^^^^^^^^^^^^^^^^
By default, ``xtl.autoproc process`` will wait until each dataset has finished processing before starting the next one.
However, if the system has adequate resources (*e.g.* in a high-performance cluster), one can perform multiple autoPROC
jobs in parallel:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --no-jobs=2

.. danger::

    Be careful when running multiple jobs in parallel, as autoPROC can be quite resource-intensive. It is recommended to
    monitor the system's resources during the run to determine the optimal number of parallel jobs.

Additionally, one can also specify the number of XDS jobs and processors for each autoPROC run, using:

.. code-block:: console

  $ xtl.autoproc process datasets.csv --xds-jobs=4 --xds-proc=8

This essentially sets the following two autoPROC parameters:

.. code-block::

  autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS=4
  autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_PROCESSORS=8

``process_wf`` command
----------------------
The ``process_wf`` command is a variation of the ``process`` command, intended to process datasets that have been
collected using the Global Phasing workflow, available at certain synchrotrons. When collecting data with the GPhL
workflow, a ``stratcal_gen.nml`` file is generated, containing information about the different sweeps, how they are
related to each other, *etc*, but most importantly (at least for ``xtl.autoproc``), the paths to the images.

While these data can be merged manually (see: `Dataset merging`_), it is recommended to make use of
the information available in the ``stratcal_gen.nml`` file. Typically, one would run the autoPROC ``process_wf`` command
instead, providing the NML file as input. The ``xtl.autoproc process_wf`` does exactly that, but also enables queuing
multiple runs with the use of a CSV file, similar to the ``xtl.autoproc process`` command.

The typical usage of the ``xtl.autoproc process_wf`` command is as follows:

.. code-block:: console

    $ xtl.autoproc process_wf datasets.csv

where ``datasets.csv`` now contains a list of paths to NML files, instead of first images, *e.g.*:

.. code-block:: csv
   :caption: ``datasets.csv``

   # nml_file
   /path/to/dataset1/stratcal_gen.nml
   /path/to/dataset2/stratcal_gen.nml
   /path/to/dataset3/stratcal_gen.nml

Although additional options can be passed along using the CSV file, do note that the NML files already contain
information about unit-cell, space group, relative crystal orientation between sweeps, *etc*.

Updating NML files
~~~~~~~~~~~~~~~~~~
The ``stratcal_gen.nml`` files are generated during data collection at the synchrotron. This means that the paths to the
collected images will follow the file structure of the light source. If the data have been transferred to a different
location (*e.g.* a local drive), then all the paths in the NML file would be invalid. However, the NML file still
contains valuable information and should be the preferred way of processing GPhL workflow data.

In order to update the image directories within an NML file, a simple command is provided:

.. code-block:: console

    $ xtl.autoproc fixnml stratcal_gen.nml --from="/synchrotron/path/to/raw_data" --to="/local/path/to/raw_data" --check

This will read the ``stratcal_gen.nml`` file, and update the ``NAME_TEMPLATE`` for each sweep, by replacing the
value of ``--from`` with the value of ``--to``. For example, if the NML file contains:

.. code-block::
    :caption: ``stratcal_gen.nml``

    &SIMCAL_SWEEP_LIST
        NAME_TEMPLATE = '/synchrotron/path/to/raw_data/datasets/dataset1/dataset1_####.cbf.gz'

then a ``stratcal_gen_updated.nml`` file will be created with:

.. code-block::
    :caption: ``stratcal_gen_updated.nml``

    &SIMCAL_SWEEP_LIST
        NAME_TEMPLATE = '/local/path/to/raw_data/datasets/dataset1/dataset1_####.cbf.gz'

The ``--check`` flag will perform a GLOB search for ``dataset1_*.cbf.gz`` within the new directory, *i.e.*
``/local/path/to/raw_data/datasets/dataset1/``, and if no files match that pattern, the user will be notified.

Multiple NML files can be updated at once, by providing a list of them as arguments to the command. At the end, an
``updated_nml.csv`` file will be saved in the current working directory, which will contain the absolute paths to the
updated NML files and can be passed directly to ``xtl.autoproc process_wf``.

.. _options-command:

``options`` command
-------------------
The ``options`` command prints a table of all supported options for dataset discovery and autoPROC configuration that
can be parsed from the CSV file.

CSV specification
-----------------
The ``datasets.csv`` file is a powerful way to fully customize the autoPROC runs on a per-dataset basis. Various options
influencing the dataset discovery or autoPROC configuration can be passed along.

.. seealso::

    See the :ref:`options command <options-command>` for more details on the available options.

When preparing the input CSV file, a few rules should be followed:

1. The first line should start with ``#`` followed by a space, and then comma-separated list of column names (without
   spaces). The order of the columns is not important.
2. If an unknown column is found in the header, it will be ignored.
3. Any subsequent lines starting with ``#`` will be treated as comments and ignored.
4. Each line should contain values for one dataset, in the same order as in the header.
5. Each line should contain the same number of values as the header, but one or more of them can be empty.
6. Each line should be terminated with a newline character.
7. Spaces in values will not be trimmed!

Taking all the above into account, a more complex CSV file might look like this:

.. code-block:: csv
   :caption: ``datasets.csv``

   # first_image,unit_cell,space_group,reference_mtz,beamline
   /path/to/dataset1/dataset1_00001.cbf.gz,78;78;37;90;90;90,P 43 21 2,,
   /path/to/dataset2/dataset2_00002.cbf.gz,,,/path/to/reference.mtz,
   /path/to/dataset3/dataset3_00003.cbf.gz,,,,PetraIIIP14

This will run the first dataset with the specified space group and unit-cell, the second dataset with a reference MTZ
file and the last one with the ``PetraIIIP14`` macro file. Notice that each line contains the same number of commas,
meaning that they all have the same number of columns. Also note that the ``unit_cell`` parameter is provided as a
semicolon-separated list of values.

Any dataset-specific options specified on the CSV file will first be merged with the global options passed along the
``xtl.autoproc process`` command,. If an option is specified both on a global and a dataset level, then the dataset one
will take precedence. For example, running the above CSV file with:

.. code-block:: console

    $ xtl.autoproc process datasets.csv --space-group="P 21"

then the first dataset will be processed with space group :math:`P 4_3 2_1 2`, while the rest with :math:`P 2_1`.

One can easily image the flexibility for fully customized runs that is provided with this architecture.

Technical documentation
-----------------------

Dataset discovery
~~~~~~~~~~~~~~~~~
The dataset discovery process is a crucial part of ``xtl.autoproc`` as it determines the output directories for each
autoPROC run. When provided with an absolute path to the first image, ``xtl.autoproc`` tries to extract three values
from that path: ``raw_data_dir``, ``dataset_dir`` and ``dataset_name``. The ``dataset_dir`` is particularly important,
because the same value will be used to determine the output path, *i.e.* ``{output_dir}/{dataset_dir}``.

Let's consider the following example:

.. code-block:: csv
    :caption: ``datasets.csv``

    # first_image
    /path/to/raw_data/datasets/dataset1/dataset1_measurement1_00001.cbf.gz

When no additional information is provided, the ``dataset_dir`` is assumed to be the parent directory of the first
image, and everything preceding that will be the ``raw_data_dir``, *i.e.*:

.. code-block:: text

    /path/to/raw_data/datasets/dataset1/dataset1_measurement1_00001.cbf.gz
    \________________________/ \______/ \___________________/
           raw_data_dir       dataset_dir    dataset_name

In this case, the output directory for the dataset will be ``{output_dir}/dataset1``, where ``output_dir`` can be
specified using the ``--out-dir`` option (default: current working directory).

However, the discovery process can be influenced by providing a raw data directory, either globally with the
``--raw-dir`` option or on a per-dataset basis with the ``raw_data_dir`` parameter on the CSV file (see:
`CSV specification`_). If, for example, the user specifies ``--raw-dir="/path/to/raw_data"``, then the same path will be
split as follows:

.. code-block:: text

    /path/to/raw_data/datasets/dataset1/dataset1_measurement1_00001.cbf.gz
    \_______________/ \_______________/ \___________________/
       raw_data_dir      dataset_dir         dataset_name

and the output directory for the dataset will be ``{output_dir}/datasets/dataset1``.

Optionally, one can specify subsequent subdirectories within the above location with the ``--out-subdir`` flag or
``output_subdir`` column in the CSV file. For example, setting ``--out-subdir=my/output`` will put the output of
autoPROC in ``{output_dir}/datasets/dataset1/my/output``.

Dataset name determination
^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to explicitly instruct autoPROC to process a specific dataset, ``xtl.autoproc`` constructs a ``-Id`` flag,
which is in the form of:

.. code-block:: text

    -Id xtl1234,/path/to/raw_data/datasets/dataset1/,dataset1_measurement1_0####.cbf.gz,1,3600
        \_____/ \_________________________________/ \_________________________________/ | \__/
        sweep_id          image_directory                      image_template           |  last_image_no
                                                                                        first_image_no

The ``sweep_id`` is a unique alphanumeric identifier (irrelevant for the user), ``image_directory`` is set to
``{raw_data_dir}/{dataset_dir}/``, while ``image_template``, ``first_image_no`` and ``last_image_no`` are required
for ``XDS.INP``. The part of the ``image_template`` preceding the ``#`` characters is called ``dataset_name`` within
``xtl.autoproc``.

Essentially, the ``dataset_name`` is the part of the image filename preceding the image number
(``dataset1_measurement1`` in the above example). To determine that value, a GLOB search is performed within
the dataset directory, the results are sorted alphabetically, and a character-by-character comparison is performed
between the first and last result, *i.e.* the first and last image (hopefully). Then ``dataset_name`` is set to the
longest common substring between the two files, *e.g.* ``dataset1_measurement1_0`` if the first and last images are
``dataset1_measurement1_00001.cbf.gz`` and ``dataset1_measurement1_03600.cbf.gz``.

In practice, there are a few more tricks in place to ensure that the dataset name determination is robust enough,
although it can never be foolproof. In case of atypical naming conventions, the automatic dataset name determination
might fail. In such cases, one can explicitly define all the above parameters in the CSV file, *e.g.*:

.. code-block:: csv
    :caption: ``datasets.csv``

    # raw_data_dir,dataset_dir,dataset_name
    /path/to/raw_data,/datasets/dataset1,dataset1_measurement1.cbf.gz

Notice that ``dataset1_measurement1.cbf.gz`` is not the same as the first image filename
(``dataset1_measurement1_00001.cbf.gz``), but rather the ``dataset_name`` and the file extension. This is enough to
determine the first image by performing a GLOB search for ``{dataset_name}*{file_extension}`` within
``{raw_data_dir}/{dataset_dir}``. The first alphabetically sorted result will be considered as the first image.

Job execution
~~~~~~~~~~~~~
For each dataset to be processed, ``xtl.autoproc`` will create a job. A job is a subprocess that runs autoPROC in the
background. To better organize the autoPROC runs, we have opted to include the input to autoPROC in a ``.dat`` macro
file, which is in turn passed to the autoPROC ``process`` command, via an intermediate shell script.

Once a job is launched, the job directory is first created within ``{output_dir}/{dataset_dir}``, typically in the form
of ``autoproc_runXX``, and then the macro file and shell script are created within that directory. A typical shell
script and macro file will look like this:

.. code-block:: bash
    :caption: ``xtl_autoPROC.sh``

    #!/bin/bash
    process -M /path/to/processed/datasets/dataset1/autoproc_run01/xtl_autoPROC.dat -d /path/to/processed/datasets/dataset1/autoproc_run01/autoproc

.. code-block::
    :caption: ``xtl_autoPROC.dat``

    # autoPROC macro file
    # Generated by xtl v.0.1.0 on 2025-12-30T19:47:35.620825
    #  user@host [distro]

    ### Dataset definitions
    # autoproc_id = xtl7156
    # no_sweeps = 1
    ## Sweep 1 [xtl7156]: dataset1_measurement1
    #   raw_data = /path/to/raw_data/datasets/dataset1/
    #   first_image = dataset1_measurement1_00001.cbf.gz
    #   image_template = dataset1_measurement1_#####.cbf.gz
    #   img_no_first = 1
    #   img_no_last = 3600
    #   idn = xtl7156,/path/to/raw_data/datasets/dataset1/,dataset1_measurement1_#####.cbf.gz,1,3600

    ### CLI arguments (including dataset definitions and macros)
    __args='-Id "xtl7156,/path/to/raw_data/datasets/dataset1/,dataset1_measurement1_#####.cbf.gz,1,3600" -B -M HighResCutOnCChalf'

    ### User parameters
    cell="79.0 79.0 37.0 90.0 90.0 90.0"
    symm="P43212"
    nres=129

    ### XDS parameters
    autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS=16
    autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_PROCESSORS=4
    ...

As you can see, the ``xtl_autoPROC.sh`` runs the autoPROC ``process`` command by passing the ``xtl_autoPROC.dat`` macro
file as input and the output directory for the autoPROC run. In turn, the macro file contains the rest of the autoPROC
parameters, as well as some debug information for the dataset discovery process of ``xtl.autoproc``.

When the ``xtl_autoPROC.sh`` script is executed as a subprocess, its standard output and error streams (and subsequently
those of the autoPROC ``process`` command) are redirected to ``xtl_autoPROC.stdout.log`` and
``xtl_autoPROC.stderr.log``, respectively, withing the job directory.

The jobs are executed asynchronously, and the main event loop ensures that only a certain number of jobs are running
simultaneously (controlled by the ``--no-jobs`` option, default: ``1``).

Post-completion tidy-up
~~~~~~~~~~~~~~~~~~~~~~~
Once the autoPROC ``process`` command exits, a few tidy-up tasks are triggered. First, ``xtl.autoproc`` will try to
determine if the autoPROC run was successful (*i.e.* whether it yielded a reflection file), by checking for the presence
of the ``staraniso_alldata-unique.mtz`` file. It then copies the following files from the autoPROC output directory to
the job directory, for easier access:

- ``summary.html``
- ``report.pdf``
- ``report_staraniso.pdf``
- ``truncate-unique.mtz``
- ``staraniso_alldata-unique.mtz``
- ``xtlXXXX.dat``

The ``xtlXXXX.dat`` file contains all the parameters that autoPROC digested from the user's input. The two MTZ files
will be prepended with the ``dataset_name``.

Finally, if the autoPROC run was deemed successful, an ``xtl_autoPROC.json`` file will also be generated within the job
directory. This JSON file combines results from ``imginfo.xml``, ``truncate.xml``, ``staraniso.xml`` and ``CORRECT.LP``,
and can be very convenient for downstream programmatic parsing of autoPROC results. A little jiffy
(``xtl.autoproc json2csv``) is provided to convert the JSON files from all jobs into a single monolithic CSV file, that
may or may not be more convenient to work with than the individual JSON files.

Advanced parametrization
~~~~~~~~~~~~~~~~~~~~~~~~
Although ``xtl.autoproc`` includes options for the most frequently used autoPROC parameters, sometimes it may not be
enough. To ensure that any possible autoPROC configuration can be launched within the provided framework, any arbitrary
parameter can be passed along to autoPROC using one of two ways.

On a global level, using the ``-x``/``--extra`` option will pass a single ``parameter=value`` pair. If more than one
parameter need to be passed along, then the ``-x`` option can be provided multiple times, *e.g.*:

.. code-block:: console

    $ xtl.autoproc process datasets.csv -x autoPROC_XdsIntegPostrefNumCycle=5 -x wave=0.9876

On a dataset level, the same can be achieved by specifying an ``extra_params`` column, which expects a
semicolon-separated list of ``parameter=value`` pairs, *e.g.*:

.. code-block:: csv
   :caption: ``datasets.csv``

   # first_image,extra_params
   /path/to/dataset1/dataset1_00001.cbf.gz,autoPROC_XdsIntegPostrefNumCycle=5;wave=0.9876
   /path/to/dataset2/dataset2_00001.cbf.gz,wave=0.9876

Internally, the provided arguments will be converted into a ``{parameter: value}`` dictionary by splitting on the ``=``
character, and then try to apply proper character escaping to the value, *e.g.* padding with double quotes if it
contains any spaces, before passing it to the autoPROC command. However, do note that no checks will be performed on the
parameter names to ensure that they are valid autoPROC parameters. This responsibility falls on the user. A list of all
the supported autoPROC parameters can be found `here <https://www.globalphasing.com/autoproc/manual/appendix1.html>`_.