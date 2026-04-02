from pathlib import Path

import typer

from xtl import settings
from xtl.cli.utilities.decorators import typer_async, attach_hook, job_options
from xtl.cli.utilities.common import get_console_options, ConsoleOptions, \
    get_job_options, JobOptions


app = typer.Typer()


@app.command('compare', help='Compare two or more SAXS datasets using datcmp')
@job_options(dependencies=['atsas'])
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_job_options, hook_output='job_options')
@typer_async
async def cli_saxs_compare(
    datafiles: list[Path] = typer.Argument(..., metavar='FILE(S)',
                                           help='Data files to compare'),
    alpha: float = typer.Option(0.01, '--alpha', '-a', min=0,
                                help='Significance level for clique search'),
    job_options: JobOptions = typer.Option(),
    console_options: ConsoleOptions = typer.Option(),
):
    import tempfile
    from xtl.cli.utilities.console import ConsoleIO
    from xtl.saxs.jobs.compare import SAXSCompareJob, SAXSCompareJobConfig

    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.report_job_options(job_options)
    settings.jobs.keep_temp = job_options.keep_temp

    job_directory = Path(tempfile.mkdtemp(prefix='xtl_saxs_compare_'))
    config = SAXSCompareJobConfig(
        job_directory=job_directory,
        files=datafiles,
        steps={
            'datcmp_batch': {
                'options': {
                    'alpha': alpha,
                }
            }
        }
    )
    with console.get_pool() as pool:
        jobs = pool.submit(SAXSCompareJob, configs=[config])
        results = await pool.launch()
    if not results[0].success:
        raise typer.Exit(code=1)

    results = results[0]

    from rich.tree import Tree

    tree = Tree('[bold]Datasets[/]')
    cliques = results.data['cliques']
    for i, lineage in enumerate(cliques):
        branch = tree.add(f'[bold green]Clique #{i + 1:,}[/]')
        for file in lineage:
            branch.add(file.name)
    console.print(tree)

    console.print(f'\nNumber of unique merging cliques: '
                  f'[dim]{len(cliques):,}[/]', highlight=False)
    console.print(f'Longest clique: [dim]#1 '
                  f'({len(cliques[0])}/{len(datafiles)} datasets)[/]', highlight=False)
