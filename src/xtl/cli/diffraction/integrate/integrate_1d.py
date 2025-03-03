from pathlib import Path

import matplotlib.pyplot as plt
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import typer

from xtl.cli.cliio import Console
from xtl.cli.diffraction.cli_utils import get_image_frames, IntegrationErrorModel, IntegrationRadialUnits


app = typer.Typer()


@app.command('1d', help='Perform 1D integration')
def cli_diffraction_integrate_1d(
        images: list[str] = typer.Argument(..., help='Images to integrate'),
        geometry: Path = typer.Option(..., '-g', '--geometry', help='Geometry .PONI file',
                                      exists=True),
        # mask: Path = typer.Option(None, '-m', '--mask', help='Mask file'),
        # Integration parameters
        points_radial: int = typer.Option(300, '-pR', '--points-radial', help='Number of points along the radial axis',
                                          min=50, rich_help_panel='Integration parameters'),
        units_radial: IntegrationRadialUnits = typer.Option(IntegrationRadialUnits.TWOTHETA_DEG.value, '-uR', '--units-radial', help='Units along the radial axis',
                                                            rich_help_panel='Integration parameters'),
        error_model: IntegrationErrorModel = typer.Option(IntegrationErrorModel.POISSON.value, '-e', '--error-model', help='Error model',
                                                          rich_help_panel='Integration parameters'),
        # Plotting parameters
        plot: bool = typer.Option(False, '-P', '--plot', help='Plot the integrated data',
                                    rich_help_panel='Plotting parameters', is_flag=True),
        xlog: bool = typer.Option(False, '--xlog', help='Use logarithmic scale for the x-axis',
                                    rich_help_panel='Plotting parameters', is_flag=True),
        ylog: bool = typer.Option(False, '--ylog', help='Use logarithmic scale for the y-axis',
                                    rich_help_panel='Plotting parameters', is_flag=True),
        # Output parameters
        output_dir: Path = typer.Option('.', '-o', '--output', help='Output directory',
                                        rich_help_panel='Output parameters'),
        include_headers: bool = typer.Option(False, '-H', '--headers', help='Include headers in the output files',
                                             rich_help_panel='Output parameters', is_flag=True),
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

    try:
        for i, image in enumerate(images):
            image.load_geometry(geometry)
    except Exception as e:
        cli.print_traceback(e)
        cli.print(f'Error: Failed to load geometry file {geometry} for image {input_images[i]}', style='red')
        raise typer.Abort()
    g = images[0].geometry
    cli.print(f'Wavelength read from geometry file: {g.get_wavelength() * 1e10:.6f} \u212B')

    try:
        for i, image in enumerate(images):
            image.initialize_azimuthal_integrator(dim=1)
    except Exception as e:
        cli.print_traceback(e)
        cli.print(f'Error: Failed to initialize azimuthal integrator for image {input_images[i]}', style='red')
        raise typer.Abort()

    integration_kwargs = {
        'points_radial': points_radial,
        'units_radial': units_radial.value,
        'error_model': error_model.value if error_model != IntegrationErrorModel.NONE else None,
    }

    output_dir = output_dir.expanduser().resolve()
    fig, ax = plt.subplots()
    xscale = 'log' if xlog else 'linear'
    yscale = 'log' if ylog else 'linear'
    try:
        with Progress(*Progress.get_default_columns(), transient=True) as progress:
            task = progress.add_task('Integrating images... ', total=len(images))
            for i, image in enumerate(images):
                # Integrate data
                image.ai1.initialize(**integration_kwargs)
                image.ai1.integrate(keep=True)

                # Save results
                if image._is_multifile:
                    filename = f'{image.file.stem}.xye'
                else:
                    filename = f'{image.file.stem}_{image.frame}.xye'
                output_file = output_dir / filename
                image.ai1.save(output_file, header=include_headers, overwrite=overwrite)
                progress.update(task, advance=1)

                if plot:
                    image.ai1.plot(ax=ax, label=filename, xscale=xscale, yscale=yscale)

    except Exception as e:
        cli.print_traceback(e)
        cli.print(f'Error: Failed to integrate image {input_images[i]}', style='red')
        raise typer.Abort()
    except FileExistsError as e:
        cli.print_traceback(e)
        cli.print(f'Error: Output file already exists', style='red')
        raise typer.Abort()

    cli.print(f'Integration complete. Results saved to: {output_dir}', style='green')

    if plot:
        ax.legend()
        plt.show()