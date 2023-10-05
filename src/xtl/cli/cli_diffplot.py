import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from rich.progress import track
from tabulate import tabulate
import typer

from xtl.cli.cliio import CliIO
from xtl.diffraction.images.images import Image
from xtl.diffraction.images.masks import detector_masks
from xtl.math import si_units


app = typer.Typer(name='diffplot', help='Plot diffraction data')


@app.command('frames', help='Plot multiple 2D images')
def cli_diffplot_frames(fname: Path = typer.Argument(metavar='FILE'),
                        frames: int = typer.Argument(1, help='No of frames to plot'),
                        log_intensities: bool = typer.Option(False, '-l', '--log', help='Logarithmic intensity scale'),
                        hot_pixel: float = typer.Option(None, '-hp', '--hot_pixel', help='Pixels with value greater '
                                                                                         'than this will be masked'),
                        low_counts: bool = typer.Option(False, '-lc', '--low_counts', help='High-contrast mode for '
                                                                                           '<5 counts'),
                        mask: str = typer.Option(None, '-m', '--mask', help='Apply detector geometry mask'),
                        save: bool = typer.Option(False, '-s', '--save', help='Export image to file'),
                        cmap: str = typer.Option('magma', '--cmap', help='Intensity colormap'),
                        cbad: str = typer.Option('white', '--cbad', help='Color for bad pixels'),
                        vmin: float = typer.Option(None, '--vmin', help='Colorscale minimum value'),
                        vmax: float = typer.Option(None, '--vmax', help='Colorscale maximum value'),
                        mask_gaps: bool = typer.Option(False, '-mg', help='Mask detector gaps'),
                        mask_frame: bool = typer.Option(False, '-mf', help='Mask detector frame'),
                        mask_double_pixels: bool = typer.Option(False, '-md', help='Mask double pixels')):
    # Check image file and format
    cli = CliIO()
    if not fname.exists():
        cli.echo(f'File {fname} does not exist', level='error')
        raise typer.Abort()
    if fname.suffix != '.h5' and frames > 1:
        cli.echo('Multiple frames are supported in .h5 images only. Ignoring option...', level='warning')
        frames = 1

    # Check detector mask
    if mask and mask not in detector_masks.keys():
        cli.echo(f'No mask available for detector {mask}. Choose one from: f{", ".join(detector_masks.keys())}')
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

        # Grab data and apply detector mask
        if mask:
            if mask_gaps is False and mask_frame is False and mask_double_pixels is False:
                mask_gaps, mask_frame, mask_double_pixels = True, True, True
            img.mask.mask_detector(mask, gaps=mask_gaps, frame=mask_frame, double_pixels=mask_double_pixels)
            data = img.data_masked
        else:
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
            m = ax.imshow(data, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True),
                          origin=img.detector_image_origin)

        # Frame numbering
        ax.text(0.95, 0.95, f'#{img.frame}', va='top', ha='right', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'linewidth': 0})

    fig.suptitle(img.file.name)
    ax.cax.colorbar(m)
    if save:
        export_file = (Path.cwd() / fname.name).with_suffix('.png')
        cli.echo(f'Saving plot to file {export_file}', level='info')
        plt.savefig(export_file, dpi=200)
    else:
        plt.show()


@app.command('powder', help='Plot multiple 1D patterns')
def cli_diffplot_powder():
    raise NotImplementedError


@app.command('stats', help='Statistics about 2D images')
def cli_diffplot_stats(fname: Path = typer.Argument(metavar='FILE'),
                       frame: int = typer.Argument(0, help='No of frame to probe'),
                       hot_pixel: float = typer.Option(None, '-hp', '--hot_pixel', help='Pixels with value greater '
                                                                                        'than this will be ignored'),
                       vmin: float = typer.Option(None, '--vmin', hidden=True, help='Colorscale minimum value'),
                       vmax: float = typer.Option(None, '--vmax', hidden=True, help='Colorscale maximum value')):
    # Check image file and format
    cli = CliIO()
    if not fname.exists():
        cli.echo(f'File {fname} does not exist', level='error')
        raise typer.Abort()

    # Read image
    img = Image()
    img.open(fname, frame=frame)
    data = img.data.astype('float')

    # Ignore criteria
    if hot_pixel:
        data[data >= hot_pixel] = np.nan
    if vmin:
        data[data < vmin] = np.nan
    if vmax:
        data[data > vmax] = np.nan

    # Report statistics
    cli.echo('I/O stats')
    stats = list()
    stats.append(['Filename', img.file])
    stats.append(['Format', img.fmt])
    stats.append(['File size', si_units(fname.stat().st_size, base=1024, digits=3, suffix='B')])
    stats.append(['Frames', img.no_frames])
    stats.append(['Current frame', img._fabio.currentframe])
    cli.echo(tabulate(stats, tablefmt='simple') + '\n')

    cli.echo('Image dimensions')
    dims = list()
    dims.append(['Dimension 1', img._fabio.shape[-1]])
    dims.append(['Dimension 2', img._fabio.shape[-2]])
    pixels_total = img._fabio.shape[-1] * img._fabio.shape[-2]
    dims.append(['Total pixels', pixels_total])
    cli.echo(tabulate(dims, tablefmt='simple') + '\n')

    cli.echo('Intensity statistics')
    ints = list()
    ints.append(['Maximum', np.nanmax(data)])
    ints.append(['Minimum', np.nanmin(data)])
    ints.append(['Mean', round(np.nanmean(data), 5)])
    ints.append(['Median', np.nanmedian(data)])
    ints.append(['Std', round(np.nanstd(data), 5)])
    ints.append(['Sum', np.nansum(data)])
    pixels_ignored = np.count_nonzero(np.isnan(data))
    ints.append(['Ignored pixels', f'{pixels_ignored} / {round(pixels_ignored/pixels_total * 100, 2)}%'])
    cli.echo(tabulate(ints, tablefmt='simple') + '\n')


@app.command('blemishes', help='Create a list of 2D detector blemishes')
def cli_diffplot_blemishes(fname: Path = typer.Argument(metavar='FILE'),
                           frame: int = typer.Argument(0, help='No of frame to probe'),
                           hot_pixel: float = typer.Option(None, '-hp', '--hot_pixel',
                                                           help='Pixels with value greater than this will be '
                                                                'considered as a blemish'),
                           cold_pixel: float = typer.Option(None, '-cp', '--cold_pixel',
                                                            help='Pixels with value less than this will be '
                                                                 'considered as a blemish'),
                           mask: str = typer.Option(None, '-m', '--mask', help='Apply detector geometry mask'),
                           include_detector: bool = typer.Option(False, '-d', '--include_detector',
                                                                 help='Include detector masked pixels as blemishes')):
    # Check image file and format
    cli = CliIO()
    if not fname.exists():
        cli.echo(f'File {fname} does not exist', level='error')
        raise typer.Abort()

    # Check detector mask
    if mask and mask not in detector_masks.keys():
        cli.echo(f'No mask available for detector {mask}. Choose one from: {", ".join(detector_masks.keys())}',
                 level='error')
        raise typer.Abort()

    # Check no of frames
    img = Image()
    img.open(fname, frame=frame)

    # Grab data and apply detector mask
    if mask:
        img.mask.mask_detector(mask, gaps=True, frame=True, double_pixels=True)
        if include_detector:
            data = img.data_masked.filled(np.nan)
        else:
            data = img.data_masked.filled(0.)
    else:
        data = img.data.astype('float')

    # Mark blemishes as NaN's
    if hot_pixel:
        data[data >= hot_pixel] = np.nan
    if cold_pixel:
        data[data <= cold_pixel] = np.nan

    blemishes = np.argwhere(np.isnan(data))  # returns list of [y, x] pairs
    blemishes = np.fliplr(blemishes)  # reverse list to [x, y]
    no_blemishes = len(blemishes)

    bfile = Path.cwd() / 'blemishes.txt'
    with open(bfile, 'w') as fp:
        fp.write(f'# Image file: {fname}\n')
        fp.write(f'# Frame: {frame}\n')
        fp.write(f'# Blemish criteria: {mask=}, {hot_pixel=}, {cold_pixel=}, {include_detector=}\n')
        fp.write(f'# No. of blemishes: {no_blemishes}\n')
        fp.write(f'# Pixel (x, y) coordinates list\n')
        fp.write('\n'.join([f'{x},{y}' for x, y in blemishes]))
    cli.echo(f'{no_blemishes} blemishes found!')
    cli.echo(f'Blemish list written to: {bfile}')


if __name__ == '__main__':
    app()
