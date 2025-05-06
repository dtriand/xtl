import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import typer

from xtl.cli.cliio import Console, epilog
from xtl.cli.diffraction.cli_utils import get_image_frames, ZScale


app = typer.Typer()


@app.command('2d', help='Plot diffraction images', epilog=epilog)
def cli_diffraction_plot_2d(
        images: list[str] = typer.Argument(..., help='Images to integrate'),
        # Plotting parameters
        zscale: ZScale = typer.Option(ZScale.LINEAR.value, '-z', '--zscale', help='Intensity scale',
                                      rich_help_panel='Plotting parameters'),
        zmin: float = typer.Option(None, '--zmin', help='Minimum value for intensities',
                                    rich_help_panel='Plotting parameters'),
        zmax: float = typer.Option(None, '--zmax', help='Maximum value for intensities',
                                    rich_help_panel='Plotting parameters'),
        normalize: bool = typer.Option(False, '-n', '--normalize', help='Normalize the intensities',
                                       rich_help_panel='Plotting parameters'),
        nstd: float = typer.Option(4.0, '-N', '--nstd', help='Number of standard deviations for normalization',
                                   rich_help_panel='Plotting parameters'),
        hot_pixels: float = typer.Option(None, '-hp', '--hot-pixels', help='Mask pixels with value greater than this',
                                         rich_help_panel='Plotting parameters'),
        # Output parameters
        save: bool = typer.Option(False, '-s', '--save', help='Export images to files',
                                  rich_help_panel='Output parameters', is_flag=True),
        save_only: bool = typer.Option(False, '-S', '--save-only', help='Only save the resulting images',
                                        rich_help_panel='Output parameters', is_flag=True),
        output_dir: Path = typer.Option('.', '-o', '--output', help='Output directory',
                                        rich_help_panel='Output parameters'),
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

    # Initialize figure
    fig = plt.figure()
    nimages = len(images)
    ncols = math.ceil(math.sqrt(nimages))
    nrows = math.ceil(nimages / ncols)
    grid = AxesGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0., share_all=True,
                    cbar_mode='single', cbar_location='right', cbar_pad=0.1, cbar_size=0.1)

    for i, image in enumerate(images):
        vmin, vmax = np.nanmin(image.data_masked), np.nanmax(image.data_masked)
        ax = grid[i]
        ax.tick_params(direction='in', color='white', bottom=True, top=True, left=True, right=True)
        if hot_pixels:
            image.mask.mask_intensity_greater_than(hot_pixels)
            if verbose:
                cli.print(f'Masked hot pixels with intensity >= {hot_pixels}', style='cyan')
        if normalize:
            mean = np.nanmean(image.data_masked)
            std = np.nanstd(image.data_masked)
            if zmin is not None and verbose:
                cli.print(f'Skipping zmin normalization: already set', style='yellow')
            else:
                vmin = mean - nstd * std
            if zmax is not None and verbose:
                cli.print(f'Skipping zmax normalization: already set', style='yellow')
            else:
                vmax = mean + nstd * std
            if verbose:
                cli.print(f'Normalization: mean={mean:.4e}, std={std:.4e}, zmin={vmin:.4e}, zmax={vmax:.4e}', style='cyan')
        else:
            if zmin is not None:
                vmin = zmin
            if zmax is not None:
                vmax = zmax


        _, _, m = image.plot(ax=ax, fig=fig, zscale=zscale.value, zmin=vmin, zmax=vmax, title='',
                             cmap='magma', bad_value_color='black')

        # Frame numbering
        if nimages > 1:
            ax.text(0.95, 0.95, f'#{i}', va='top', ha='right', transform=ax.transAxes,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'linewidth': 0})
            if i == nimages - 1:
                ax.cax.colorbar(m)
                if verbose:
                    cli.print(f'Colorbar: zmin={vmin:.4e}, zmax={vmax:.4e}', style='cyan')
        else:
            ax.set_title(f'{image.file.name}:{image.frame}')
            ax.cax.colorbar(m)
            if verbose:
                cli.print(f'Colorbar: zmin={vmin:.4e}, zmax={vmax:.4e}', style='cyan')

        # Save image
        if save:
            output_image = output_dir.expanduser().resolve() / f'{image.file.stem}.png'
            if output_image.exists() and not overwrite:
                cli.print(f'Error: File {output_image} already exists, use --force to overwrite', style='red')
                raise typer.Abort()

            output_image.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_image, dpi=300)

    # Plot image
    if not save_only:
        plt.show()

    # Clean up figure
    plt.close(fig)
