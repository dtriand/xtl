from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import typer

from xtl.cli.cliio import Console, epilog
from xtl.cli.diffraction.cli_utils import get_image_frames


app = typer.Typer()


@app.command('sum', help='Sum or average diffraction images', epilog=epilog)
def cli_diffraction_sum(
        images: list[str] = typer.Argument(..., help='Images to integrate'),
        # Summation parameters
        average: bool = typer.Option(False, '-a', '--average', help='Average the images instead',
                                     rich_help_panel='Summation parameters', is_flag=True),
        # Plotting parameters
        plot: bool = typer.Option(False, '-P', '--plot', help='Plot the summed image',
                                    rich_help_panel='Plotting parameters', is_flag=True),
        # Output parameters
        output_dir: Path = typer.Option('.', '-o', '--output', help='Output directory',
                                        rich_help_panel='Output parameters'),
        # include_headers: bool = typer.Option(False, '-H', '--headers', help='Include headers in the output files',
        #                                      rich_help_panel='Output parameters', is_flag=True),
        overwrite: bool = typer.Option(False, '-f', '--force', help='Overwrite existing files',
                                        rich_help_panel='Output parameters', is_flag=True),
        # Debugging
        verbose: int = typer.Option(0, '-v', '--verbose', count=True, help='Print additional information',
                                    rich_help_panel='Debugging'),
        debug: bool = typer.Option(False, '--debug', hidden=True, help='Print debug information',
                                   rich_help_panel='Debugging'),
):
    cli = Console(verbose=verbose, debug=debug)
    input_images = images

    try:
        images = get_image_frames(input_images)
    except ValueError as e:
        cli.print_traceback(e)
        cli.print(f'Error: Failed to read all images', style='red')
        raise typer.Abort()

    shape = images[0].data.shape
    for i, image in enumerate(images[1:]):
        if image.data.shape != shape:
            cli.print(f'Error: Image {input_images[i+1]} has a different shape than the first image', style='red')
            raise typer.Abort()

    data = images[0].data
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(),
                  MofNCompleteColumn(), transient=True, console=cli) as progress:
        task = progress.add_task('Summing images...', total=len(images))
        for img in images[1:]:
            data += img.data
            progress.advance(task)

    if average:
        data /= len(images)

    # Save summed image
    if average:
        output_image = (output_dir / f'averaged.npy').resolve()
    else:
        output_image = (output_dir / f'summed.npy').resolve()
    if output_image.exists() and not overwrite:
        cli.print(f'Error: File {output_image} already exists, use --force to overwrite', style='red')
        raise typer.Abort()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_image, data)
    cli.print(f'Saved resulting image to {output_image}')

    if plot:
        fig, ax = plt.subplots()
        m = ax.imshow(data, origin='lower')
        plt.colorbar(m, ax=ax)
        plt.savefig(str(output_image.with_suffix('.png')), dpi=300)
        plt.show()
