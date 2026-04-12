from pathlib import Path

import typer

from xtl.cli.utilities.decorators import typer_async, attach_hook, jobs_depend_on
from xtl.cli.utilities.common import get_console_options, ConsoleOptions, \
    get_job_options, JobOptions
from xtl.cli.utilities import epilog


app = typer.Typer()


@app.command('compare', help='Compare two or more SAXS datasets using datcmp', epilog=epilog)
@jobs_depend_on('atsas')
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_job_options, hook_output='job_options')
@typer_async
async def cli_saxs_compare(
    datafiles: list[Path] = \
            typer.Argument(
                ...,
                metavar='FILE(S)',
                help='Data files to compare'
            ),
    alpha: float = \
            typer.Option(
                0.01, '--alpha', '-a',
                min=0, max=1,
                help='Significance level for clique search'
            ),
    csv_output: bool = \
            typer.Option(
                False, '--csv',
                help='Output results as CSV instead of a tree view'
            ),
    job_options: JobOptions = typer.Option(),
    console_options: ConsoleOptions = typer.Option(),
):
    import tempfile
    from xtl.tui.console import ConsoleIO
    from xtl.saxs.jobs.compare import SAXSCompareJob, SAXSCompareJobConfig

    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.apply_job_options(job_options)

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
    async with console.get_pool('simple', max_jobs=5) as pool:
        jobs = pool.submit(SAXSCompareJob, configs=config)
        results = await pool.launch('all')

    results = results[0]
    if not results or not results.success:
        console.print(f'The job completed unsuccessfully. '
                      f'Inspect job output: {job_directory}', style='bold red')
        raise typer.Exit(code=1)

    cliques = results.data['cliques']
    if csv_output:
        from xtl.common.tables import Table

        no_columns = len(cliques[0]) + 1
        table = Table(headers=['Clique', 'Files'] + [''] * (no_columns - 1))
        for i, clique in enumerate(cliques):
            if len(clique) < no_columns:
                clique += [''] * (no_columns - len(clique))
            table.add_row([i + 1] + clique)

        console.print(table.to_csv(header_char='# '), highlight=False)
    else:
        from rich.tree import Tree

        tree = Tree('[bold]Datasets[/]')
        for i, lineage in enumerate(cliques):
            branch = tree.add(f'[bold green]Clique #{i + 1:,}[/]')
            for file in lineage:
                branch.add(file.name)
        console.print(tree)

        console.print(f'\nNumber of unique merging cliques: '
                      f'[dim]{len(cliques):,}[/]', highlight=False)
        console.print(f'Longest clique: [dim]#1 '
                      f'({len(cliques[0])}/{len(datafiles)} datasets)[/]', highlight=False)

    return
