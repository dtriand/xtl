from pathlib import Path

import typer

from xtl.cli.utilities.decorators import typer_async, attach_hook, jobs_depend_on
from xtl.cli.utilities.common import get_console_options, ConsoleOptions, \
    get_job_options, JobOptions
from xtl.cli.utilities import epilog
from xtl.cli.utilities.io import resolve_paths_or_glob


app = typer.Typer()


@app.command('subtract', short_help='Perform background subtractions', epilog=epilog)
@jobs_depend_on('atsas')
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_job_options, hook_output='job_options')
@typer_async
async def cli_saxs_subtract(
    datafiles: list[Path] = \
            typer.Argument(
                ...,
                metavar='FILE(S)', show_default=False,
                help='Data files to subtract from'
            ),
    backgrounds: list[Path] = \
            typer.Option(
                ..., '--background', '-b',
                show_default=False,
                callback=resolve_paths_or_glob,  # accept paths or glob expressions
                help='Background dataset(s) to subtract from input datasets'
            ),
    output_directory: Path = \
            typer.Option(
                Path.cwd(), '--output', '-o',
                show_default='cwd',
                help='Directory for subtracted datasets'
            ),
    job_options: JobOptions = typer.Option(),
    console_options: ConsoleOptions = typer.Option(),
):
    """
    Subtract a background from one or more SAXS datasets using
    [magenta]ATSAS [link=https://biosaxs-com.github.io/atsas/latest/manuals/datop.html]datop[/].

    \b
    USAGE
    [green]>[/] xtl.saxs subtract sample.dat -b background.dat

    [i]Subtract the same background from multiple datasets[/i]
    [green]>[/] xtl.saxs subtract sample1.dat sample2.dat -b background.dat

    [i]Subtract a different background from each dataset[/i]
    [green]>[/] xtl.saxs subtract sample1.dat -b background1.dat sample2.dat -b background2.dat
    [green]>[/] xtl.saxs subtract sample1.dat sample2.dat --background="background*.dat"
    """
    import tempfile
    from xtl import settings
    from xtl.tui.console import ConsoleIO
    from xtl.scattering.jobs.operations import SAXSSubtractJob, SAXSSubtractJobConfig

    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.apply_job_options(job_options)

    if len(backgrounds) == 1:
        backgrounds = [backgrounds[0]] * len(datafiles)
    elif len(backgrounds) != len(datafiles):
        console.print('The number of background files does not match the number of datasets', style='red')
        raise typer.Exit()

    configs = [
        SAXSSubtractJobConfig(
            job_directory=Path(tempfile.mkdtemp(prefix='xtl_saxs_subtract_')),
            input=datafile.absolute(),
            modifier=background.absolute(),
            output=output_directory / f'subtracted_{datafile.stem}_minus_{background.stem}{datafile.suffix}',
        )
        for datafile, background in zip(datafiles, backgrounds)
    ]

    # Spawn one batch job per available core
    async with console.get_pool('async', max_jobs=settings.jobs.resources.max_processes) as pool:
        jobs = pool.submit(SAXSSubtractJob, configs=configs)
        results = await pool.launch('all')

    for i, result in enumerate(results):
        if not result or not result.success:
            dataset = configs[i].input
            console.print(f'Subtraction failed for dataset: {dataset}', style='red', highlight=False)
    else:
        console.print(f'Subtracted {len(datafiles)} dataset{"s" if len(datafiles) > 1 else ""} '
                      f'in: {output_directory.absolute()}',
                      highlight=False)
    return
