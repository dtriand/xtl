from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import typer

from xtl.cli.cliio import Console, epilog
from xtl.cli.utils import Timer
from xtl.cli.diffraction.cli_utils import get_image_frames, IntegrationErrorModel, IntegrationRadialUnits
from xtl.diffraction.images.correlators import AzimuthalCrossCorrelatorQQ_1
from xtl.exceptions.utils import Catcher

import warnings


app = typer.Typer()


@app.command('qq', help='Calculate CCF within the same Q vector', epilog=epilog)
def cli_diffraction_correlate_qq(
        images: list[str] = typer.Argument(..., help='Images to process'),
        geometry: Path = typer.Option(..., '-g', '--geometry', help='Geometry .PONI file',
                                      exists=True),
        # mask: Path = typer.Option(None, '-m', '--mask', help='Mask file'),
        # blemishes: Path = typer.Option(None, '-b', '--blemishes', help='Blemishes file'),
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

    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback) as catcher:
        images = get_image_frames(input_images)
    if catcher.raised:
        cli.print(f'Error: Failed to read all images', style='red')
        raise typer.Abort()

    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback) as catcher:
        for i, image in enumerate(images):
            image.load_geometry(geometry)
    if catcher.raised:
        cli.print(f'Error: Failed to load geometry file {geometry} for image {input_images[i]}', style='red')
        raise typer.Abort()
    g = images[0].geometry
    cli.print(f'Wavelength read from geometry file: {g.get_wavelength() * 1e10:.6f} \u212B')

    ### Copied from xcca.py, minor modifications to make it work
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format units
    r_repr = '2\u03b8' if units_radial == IntegrationRadialUnits.TWOTHETA_DEG else 'q'

    # Integration parameters
    integration_kwargs = {
        'points_radial': points_radial,
        'units_radial': units_radial.value,
        'error_model': error_model.value if error_model != IntegrationErrorModel.NONE else None,
    }

    with (Catcher(echo_func=cli.print, traceback_func=cli.print_traceback),
          Progress(SpinnerColumn(), *Progress.get_default_columns(),
                   TimeElapsedColumn(), MofNCompleteColumn(),
                   transient=True, console=cli) as progress):
        task = progress.add_task('Calculating CCFs...', total=len(images))
        for img in images:
            dataset_name = img.file.stem.replace('_master', '')

            # Calculate 1D azimuthal integration
            if verbose:
                progress.console.print(f'Performing 1D azimuthal integration with '
                                       f'{points_radial} {r_repr} points...')
            with Timer(silent=verbose<=1, echo_func=progress.console.print):
                ai1 = img.initialize_azimuthal_integrator(dim=1)
                ai1.initialize(**integration_kwargs)
                ai1.integrate(keep=True)

            ai1_file = output_dir / f'{dataset_name}_1D.xye'
            ai1.save(ai1_file, overwrite=overwrite)
            if verbose:
                progress.console.print(f'Saved 1D integration results to {ai1_file}')

            # Calculate CCF
            if verbose:
                progress.console.print(f'Calculating CCF over {points_radial}\u00d7{points_azimuthal} '
                                       f'{r_repr}\u00d7\u03c7 points...')
            with Timer(silent=verbose<=1, echo_func=progress.console.print):
                accf = AzimuthalCrossCorrelatorQQ_1(img)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    accf.correlate(points_radial=points_radial, points_azimuthal=points_azimuthal,
                                  units_radial=units_radial.value, method=0)

            ccf_file = output_dir / f'{dataset_name}_ccf.npx'
            accf.save(ccf_file, overwrite=overwrite)
            if verbose:
                progress.console.print(f'Saved CCF results to {ccf_file}')

            # Save 2D azimuthal integration
            ai2_file = output_dir / f'{dataset_name}_2D.npx'
            img.ai2.save(ai2_file, overwrite=overwrite)
            if verbose:
                progress.console.print(f'Saved 2D integration results to {ai2_file}')

            # Prepare plots
            fig = plt.figure('XCCA overview', figsize=(16 / 1.2, 9 / 1.2))
            gs0 = fig.add_gridspec(1, 2, wspace=0.2,
                                   width_ratios=[1.2, 2])  # outer grid (1x2)
            gs1 = gs0[1].subgridspec(2, 2, wspace=0.3, hspace=0.1,
                                     height_ratios=[3, 2])  # inner grid (2x2)
            ax0 = fig.add_subplot(gs0[0, 0])  # speckle pattern
            ax1 = fig.add_subplot(gs1[0, 0])  # cake plot
            ax2 = fig.add_subplot(gs1[1, 0], sharex=ax1)  # 1D azimuthal integration
            ax3 = fig.add_subplot(gs1[0, 1])  # 2D CCF
            ax4 = fig.add_subplot(gs1[1, 1], sharex=ax3)  # Average CCF

            ax0.tick_params(direction='in', color='white', bottom=True, top=True, left=True, right=True)
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(direction='in', bottom=True, left=True)

            norm = 'linear'

            # Speckle pattern
            _, _, m0 = img.plot(ax=ax0, fig=fig, apply_mask=True, overlay_mask=True,
                                title='Speckle pattern', zscale=norm, cmap='magma')
            fig.colorbar(m0, location='bottom', pad=0.07)
            ax0.text(0.5, -0.3, 'Intensity (arbitrary units)', va='bottom', ha='center',
                     transform=ax0.transAxes, bbox={'alpha': 0.})

            # 2D integration
            _, _, m1 = img.ai2.plot(ax=ax1, fig=fig, title='Cake projection', zscale=norm,
                                    cmap='magma')
            fig.colorbar(m1, location='bottom')
            ax1.text(-0.05, -0.276, 'Int.', va='bottom', ha='right', transform=ax1.transAxes,
                     bbox={'alpha': 0.})

            # 1D integration
            img.ai1.plot(ax=ax2, fig=fig, line_color='xkcd:plum')
            ax2.set_title('Azimuthal integration')

            # CCF
            m = np.nanmean(accf.ccf)
            s = np.nanstd(accf.ccf)
            nstd = 1
            vmin = m - nstd * s
            vmax = m + nstd * s
            _, _, m3 = accf.plot(ax=ax3, fig=fig, zmin=vmin, zmax=vmax, zscale='symlog',
                                 cmap='Spectral')
            fig.colorbar(m3, location='bottom')
            ax3.text(-0.05, -0.276, 'CCF', va='bottom', ha='right', transform=ax3.transAxes,
                     bbox={'alpha': 0.})

            # Average CCF
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                ccf_mean = np.nanmean(accf.ccf, axis=0)
            ax4.plot(img.ai1.results.radial, ccf_mean, color='xkcd:plum')
            ax4.set_xlabel(accf.units_radial_repr)
            ax4.set_ylabel('\u27e8CCF\u27e9$_{\u0394}$')
            ax4.set_title('Average CCF')

            fig.suptitle(f'{img.file.name} frame #{img.frame}', y=0.95)

            # Save plot
            fig_file = output_dir / f'{dataset_name}_overview.png'
            fig.savefig(fig_file, dpi=600, bbox_inches='tight')
            progress.console.print(f'Saved overview plot to {fig_file}')

            plt.close(fig)

            # Delete integrators and image to free up memory
            #  The correlator gets overwritten on the next iteration
            del img.ai1
            del img.ai2
            del img

            progress.advance(task)
