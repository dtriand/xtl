from pathlib import Path

import typer

from xtl import settings
from xtl.cli.utilities.decorators import typer_async, attach_hook, job_options
from xtl.cli.utilities.common import get_console_options, ConsoleOptions, \
    get_automate_options, AutomateOptions, CPU_CORES
from xtl.saxs.jobs.atsas_utils import DatcmpMode, DatcmpTest, DatcmpAdjustment


app = typer.Typer()


@app.command('compare', help='Compare two or more SAXS datasets using datcmp')
@job_options(dependencies=['atsas'])
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_automate_options, hook_output='automate_options')
@typer_async
async def cli_saxs_compare(
    datafiles: list[Path] = typer.Argument(..., metavar='FILE(S)',
                                           help='Data files to compare'),
    test: DatcmpTest = typer.Option(DatcmpTest.CORMAP, '--test', '-t',
                                    help='Test name'),
    adjustment: DatcmpAdjustment = typer.Option(DatcmpAdjustment.FWER, '--adjust',
                                                help='Adjustment method'),
    alpha: float = typer.Option(0.01, '--alpha', '-a', min=0,
                                help='Significance level for clique search'),
    max_jobs: int = typer.Option(CPU_CORES * 10, '--max-jobs', min=0,
                                 help='Maximum number of concurrent jobs',
                                 rich_help_panel='Parallelization'),
    automate_options: AutomateOptions = typer.Option(),
    console_options: ConsoleOptions = typer.Option(),
):
    import tempfile
    from xtl.cli.utilities.console import ConsoleIO
    from xtl.saxs.jobs.atsas_utils import DatcmpOptions
    from xtl.saxs.jobs.compare import SAXSCompareTreeJob, SAXSCompareTreeJobConfig

    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.report_automate(automate_options)
    settings.automate.keep_temp = automate_options.keep_temp

    job_directory = Path(tempfile.mkdtemp(prefix='xtl_saxs_compare_'))
    if console.verbose:
        console.print(f'Job directory: [dim]{job_directory}[/]')
    config = SAXSCompareTreeJobConfig(
        job_directory=job_directory,
        files=datafiles,
        batch=automate_options.get_batch_config(),
        datcmp=DatcmpOptions(
            test=test,
            adjust=adjustment,
            alpha=alpha,
            mode=DatcmpMode.PAIRWISE,
        ),
        max_jobs=max_jobs,
    )
    if automate_options.compute_site == 'modules':
        config._include_default_dependencies = False

    with console.get_pool() as pool:
        jobs = pool.submit(SAXSCompareTreeJob, configs=[config])
        results = await pool.launch()
    if not results[0].success:
        raise typer.Exit(code=1)

    results = results[0]

    from rich.tree import Tree

    tree = Tree('[bold]Datasets[/]')
    for i, lineage in enumerate(results.data.lineages):
        branch = tree.add(f'[bold green]Lineage #{i + 1}[/]')
        for file in lineage:
            branch.add(str(file))
    console.print(tree)

    max_len = 1
    longest_lineage = None
    for i, lineage in enumerate(results.data.lineages):
        if len(lineage) > max_len:
            max_len = len(lineage)
            longest_lineage = i + 1
    console.print(f'\nNumber of unique merging lineages: '
                  f'[dim]{len(results.data.lineages)}[/]', highlight=False)
    console.print(f'Longest lineage: [dim]#{longest_lineage} '
                  f'({max_len}/{len(datafiles)} datasets)[/]', highlight=False)
