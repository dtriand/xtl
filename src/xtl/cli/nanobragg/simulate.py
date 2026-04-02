import asyncio
import shutil
from pathlib import Path
from typing import Annotated

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import typer

from xtl.cli.cliio import epilog
from xtl.cli.utilities.decorators import typer_async, attach_hook, job_options
from xtl.cli.utilities.common import get_console_options, ConsoleOptions, get_job_options, JobOptions

app = typer.Typer()


@app.command('simulate', short_help='Run a nanoBragg simulation', epilog=epilog)
@job_options(dependencies=['easybragg'])
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_job_options, hook_output='job_options')
@typer_async
async def cli_nanobragg_simulate(
        options_file: Path = \
                typer.Argument(
                    ...,
                    exists=True,
                    help='Path to the nanoBragg options file',
                ),
        mtz: Path = \
                typer.Option(
                    None, '-r', '--reflections',
                    exists=True,
                    rich_help_panel='Simulation',
                    help='Path to the reflections file (overrides the one from the config file)'
                ),
        use_gpu: bool = \
                typer.Option(
                    False, '--gpu',
                    is_flag=True,
                    rich_help_panel='Simulation',
                    help='Use GPU for simulation (if available)'
                ),
        output_dir: Path = \
                typer.Option(
                    '.', '-o', '--output',
                    exists=False,
                    rich_help_panel='Output',
                    help='Path to the output directory'
                ),
        force: bool = \
                typer.Option(
                    False, '-f', '--force',
                    is_flag=True,
                    rich_help_panel='Output',
                    help='Force overwrite of output directory if it already exists'
                ),
        job_options: JobOptions = typer.Option(),
        console_options: ConsoleOptions = typer.Option(),
):
    """
    Run a nanoBragg simulation based on the provided [magenta]options.json[/] file.

    USAGE
        xtl.nanobragg simulate options.json

        [i]Override reflections file[/i]
        xtl.nanobragg simulate options.json -r reflections.mtz

        [i]Toggle GPU acceleration[/i]
        xtl.nanobragg simulate options.json --gpu
    """
    from xtl.cli.utilities.console import ConsoleIO
    from xtl.nanobragg.jobs.nanobragg import NanoBraggJob, NanoBraggJobConfig
    from xtl.nanobragg.config import NanoBraggOptions

    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.report_job_options(job_options)

    if options_file.suffix not in ['.json', '.toml']:
        console.print('Options file must be JSON or TOML', style='red')
        raise typer.Abort()

    try:
        if options_file.suffix == '.json':
            options = NanoBraggOptions.from_json(options_file)
        elif options_file.suffix == '.toml':
            options = NanoBraggOptions.from_toml(options_file)
    except Exception as e:
        console.print_traceback(e)
        console.print(f'Error: Failed to read config file {options_file}', style='red')
        raise typer.Abort()

    if mtz:
        mtz = mtz.resolve()
        console.print(f'Using reflections from: [dim]{mtz}[/dim]')
        options.structure.reflections = mtz

    output_dir = output_dir.resolve()
    if output_dir.exists():
        if force:
            console.print(f'Overwriting existing output directory: [dim]{output_dir}[/dim]', style='yellow')
            shutil.rmtree(output_dir)
        else:
            console.print(f'Output directory already exists: {output_dir}\nUse --force to overwrite.', style='red')
            raise typer.Abort()

    # TODO: Propagate job_options to NanoBraggJob
    config = NanoBraggJobConfig(
        job_directory=output_dir,
        options=options,
        steps={
            'nanobragg_batch': {
                'use_gpu': use_gpu,
                'debug': console.debug,
                }
        }
    )

    with console.get_pool() as pool:
        jobs = pool.submit(NanoBraggJob, configs=[config])
        # TODO: Progress tracking for nanoBragg job
        results = await pool.launch()
    if not results[0].success:
        job = jobs[0]
        console.print(f'Simulation failed, look at logs for more details: \n'
                      f'- STDOUT: {job.config.steps["nanobragg_batch"].stdout}\n'
                      f'- STDERR: {job.config.steps["nanobragg_batch"].stderr}', style='red')
        raise typer.Exit(code=1)
    console.print(f'Simulation completed successfully, images saved to: [dim]{output_dir}[/dim]', style='green')
