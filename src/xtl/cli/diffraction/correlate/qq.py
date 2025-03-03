from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import typer

from xtl.cli.cliio import Console
from xtl.cli.diffraction.cli_utils import get_image_frames, IntegrationErrorModel, IntegrationRadialUnits
from xtl.diffraction.images.correlators import AzimuthalCrossCorrelatorQQ_1

from datetime import datetime
import warnings
from xtl.math import si_units


app = typer.Typer()


@app.command('qq', help='Calculate CCF within the same Q vector')
def cli_diffraction_correlate_qq(
images: list[str] = typer.Argument(..., help='Images to integrate'),
        geometry: Path = typer.Option(..., '-g', '--geometry', help='Geometry .PONI file',
                                      exists=True),
        # mask: Path = typer.Option(None, '-m', '--mask', help='Mask file'),
        # Integration parameters
        points_radial: int = typer.Option(300, '-pR', '--points-radial', help='Number of points along the radial axis',
                                          min=50, rich_help_panel='Integration parameters'),
        units_radial: IntegrationRadialUnits = typer.Option(IntegrationRadialUnits.TWOTHETA_DEG.value, '-uR', '--units-radial', help='Units along the radial axis',
                                                            rich_help_panel='Integration parameters'),
        points_azimuthal: int = typer.Option(360, '-pA', '--points-azimuthal', help='Number of points along the azimuthal axis',
                                             min=50, rich_help_panel='Integration parameters'),
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
                                   rich_help_panel='Debugging')
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

    ### Copied from xcca.py, minor modifications to make it work

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(),
                  MofNCompleteColumn(), transient=True, console=cli) as progress:
        task = progress.add_task('Calculating CCFs...', total=len(images))
        for img in images:
            dataset_name = img.file.stem.replace('_master', '')

            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            # Calculate 1D azimuthal integration
            progress.console.print(f'Performing 1D azimuthal integration with {points_radial} 2\u03b8 points... ', end='')
            t0 = datetime.now()
            ai1 = img.initialize_azimuthal_integrator(dim=1)
            ai1.initialize(points_radial=points_radial)
            ai1.integrate()
            t1 = datetime.now()
            t = (t1 - t0).total_seconds()
            progress.console.print(f'Completed in {si_units(t, suffix="s", digits=3)}')
            ai1_file = output_dir / f'{dataset_name}_1D.xye'
            ai1.save(ai1_file, overwrite=True)
            progress.console.print(f'Saved 1D integration results to {ai1_file}')

            # Calculate CCF
            progress.console.print(f'Calculating CCF over {points_radial}\u00d7{points_azimuthal} 2\u03b8\u00d7\u03c7 points... ', end='')
            t0 = datetime.now()
            ccf = AzimuthalCrossCorrelatorQQ_1(img)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                ccf.correlate(points_radial=points_radial, points_azimuthal=points_azimuthal, method=0)
            xcca = ccf.ccf
            t1 = datetime.now()
            t = (t1 - t0).total_seconds()
            progress.console.print(f'Completed in {si_units(t, suffix="s", digits=3)}')
            ccf_file = output_dir / f'{dataset_name}_ccf.npx'
            ccf.save(ccf_file, overwrite=True)
            progress.console.print(f'Saved CCF results to {ccf_file}')

            # Save 2D azimuthal integration
            ai2_file = output_dir / f'{dataset_name}_2D.npx'
            img.ai2.save(ai2_file, overwrite=True)
            progress.console.print(f'Saved 2D integration results to {ai2_file}')

            # Prepare plots
            fig = plt.figure('XCCA overview', figsize=(16 / 1.2, 9 / 1.2))
            gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.3)
            ax0 = fig.add_subplot(gs[:, 0])  # speckle pattern
            ax1 = fig.add_subplot(gs[0, 1])  # cake plot
            ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)  # 1D azimuthal integration
            ax3 = fig.add_subplot(gs[0, 2])  # 2D CCF
            ax4 = fig.add_subplot(gs[1, 2], sharex=ax3)  # Average CCF
            norm = 'log'

            # Speckle pattern
            _, _, m0 = img.plot(ax=ax0, fig=fig, apply_mask=True, overlay_mask=True,
                                title='Speckle pattern', zscale=norm)
            fig.colorbar(m0, location='bottom')

            # 2D integration
            _, _, m1 = img.ai2.plot(ax=ax1, fig=fig, title='Cake projection', zscale=norm)
            fig.colorbar(m1)

            # 1D integration
            img.ai1.plot(ax=ax2, fig=fig)
            ax2.set_title('Azimuthal integration')

            # CCF
            m = np.nanmean(xcca)
            s = np.nanstd(xcca)
            nstd = 1
            vmin = m - nstd * s
            vmax = m + nstd * s
            _, _, m3 = ccf.plot(ax=ax3, fig=fig, zmin=vmin, zmax=vmax, zscale='symlog')
            fig.colorbar(m3)

            # Average CCF
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                ccf_mean = np.nanmean(xcca, axis=0)
            ax4.plot(img.ai1.results.radial, ccf_mean)
            ax4.set_xlabel('Radial angle / 2\u03b8 (\u00b0)')
            ax4.set_ylabel('\u27e8CCF\u27e9$_{\u03c7}$')
            ax4.set_title('Average CCF')

            fig.suptitle(f'{img.file.name} frame #{img.frame}')

            # Save plot
            fig_file = output_dir / f'{dataset_name}_overview.png'
            fig.savefig(fig_file, dpi=600)
            progress.console.print(f'Saved overview plot to {fig_file}')

            plt.close(fig)

            progress.advance(task)
