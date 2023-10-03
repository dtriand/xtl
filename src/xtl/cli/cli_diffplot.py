import copy
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from rich.progress import track
import typer

from xtl.cli.cliio import CliIO
from xtl.diffraction.images.images import Image


app = typer.Typer(name='diffplot', help='Plot diffraction data')


@app.command('frames', help='Plot several 2D frames')
def cli_diffplot_frames(fname: Path = typer.Argument(metavar='FILE'),
                        frames: int = typer.Argument(1, help='No of frames to plot'),
                        log_intensities: bool = typer.Option(False, '-l', '--log', help='Logarithmic intensity scale'),
                        hot_pixel: float = typer.Option(None, '-hp', '--hot_pixel', help='Pixels with value greater '
                                                                                         'than this will be masked'),
                        low_counts: bool = typer.Option(False, '-lc', '--low_counts', help='High-contrast mode for '
                                                                                           '<5 counts'),
                        cmap: str = typer.Option('magma', '--cmap', hidden=True, help='Intensity colormap'),
                        cbad: str = typer.Option('white', '--cbad', hidden=True, help='Color for bad pixels'),
                        vmin: float = typer.Option(None, '--vmin', hidden=True, help='Colorscale minimum value'),
                        vmax: float = typer.Option(None, '--vmax', hidden=True, help='Colorscale maximum value')):
    # Check image file and format
    cli = CliIO()
    if not fname.exists():
        cli.echo(f'File {fname} does not exist', level='error')
        raise typer.Abort()
    if fname.suffix != '.h5':
        cli.echo('Only .h5 images are supported!', level='error')
        raise typer.Abort()

    # Initialize figure
    fig = plt.figure(figsize=(6.4, 6.8))
    num = math.ceil(math.sqrt(frames))
    grid = AxesGrid(fig, 111, nrows_ncols=(num, num), axes_pad=0, share_all=True, cbar_mode='single',
                    cbar_location='right', cbar_pad=0.1, cbar_size=0.1)

    # Check no of frames
    img = Image()
    img.open(fname, frame=0)
    if img.no_frames < frames:
        cli.echo(f'File contains only {img.no_frames} frames', level='error')
        raise typer.Abort()

    # Initialize colormap
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad(color=cbad, alpha=1.0)

    # Iterate over frames
    for i, frame in enumerate(track(range(0, frames), description='Processing frames...')):
        if i != 0:
            img.next_frame()

        ax = grid[i]
        data = img.data.astype('float')

        # Plotting options
        if low_counts:
            vmax = 5.
        elif hot_pixel:
            data[data >= hot_pixel] = np.nan

        # Plot frame
        if log_intensities:
            m = ax.imshow(data, cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True),
                          origin=img.detector_image_origin)
        else:
            m = ax.imshow(data, cmap=cmap, origin=img.detector_image_origin)

        # Frame numbering
        ax.text(0.95, 0.95, f'#{img.frame}', va='top', ha='right', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'linewidth': 0})

    fig.suptitle(img.file.name)
    ax.cax.colorbar(m)
    plt.show()


@app.command('powder', help='Plot several 1D patterns')
def cli_diffplot_powder():
    raise NotImplementedError


if __name__ == '__main__':
    app()
