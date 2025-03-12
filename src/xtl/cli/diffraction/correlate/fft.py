from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import numpy as np
import typer

from xtl.cli.cliio import Console, epilog
from xtl.cli.utils import Timer
from xtl.cli.diffraction.cli_utils import get_geometry_from_header, get_radial_units_from_header, ZScale
from xtl.exceptions.utils import Catcher
from xtl.files.npx import NpxFile
from xtl.units.crystallography.radial import RadialUnitType, RadialValue


app = typer.Typer()


@app.command('fft', help='Calculate the distribution of Fourier components', epilog=epilog)
def cli_diffraction_correlate_fft(
        ccf_file: Path = typer.Argument(..., metavar='ccf.npx', help='CCF file to analyze'),
        ai2_file: Path = typer.Argument(None, metavar='ai2.npx', help='Azimuthal integration file'),
        # Selection parameters
        selection_2theta: float = typer.Option(None, '-t', '--2theta', help='2\u03b8 angle to inspect (in \u00b0)',
                                               rich_help_panel='Selection parameters'),
        selection_q: float = typer.Option(None, '-q', '--q', help='Q value to inspect (in nm\u207B\u00B9)',
                                          rich_help_panel='Selection parameters'),
        # FFT parameters
        no_coeffs: int = typer.Option(24, '-n', '--no-coeffs', help='Number of Fourier components to display',
                                      rich_help_panel='FFT parameters'),
        fill_value: float = typer.Option(None, '--fill', help='Fill NaN values in CCF with a given value',
                                         rich_help_panel='FFT parameters'),
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
        # polar_plots: bool = typer.Option(False, '--polar', help='Plot CCF and azimuthal integration in polar coordinates',
        #                                  rich_help_panel='Plotting parameters'),
        # Debugging
        verbose: int = typer.Option(0, '-v', '--verbose', count=True, help='Print additional information',
                                    rich_help_panel='Debugging'),
        debug: bool = typer.Option(False, '--debug', hidden=True, help='Print debug information',
                                   rich_help_panel='Debugging')
):
    cli = Console(verbose=verbose, debug=debug)

    # Check if at least one selection parameter is provided
    if selection_2theta is None and selection_q is None:
        cli.print('Select a 2\u03b8 angle or Q value to calculate FFT (--2theta, --q)', style='red')
        raise typer.Abort()
    elif selection_2theta is not None and selection_q is not None:
        cli.print('Select only one parameter to calculate FFT (--2theta, --q)', style='red')
        raise typer.Abort()

    if selection_2theta is not None:
        selection = RadialValue(value=selection_2theta, type=RadialUnitType.TWOTHETA_DEG)
    else:
        selection = RadialValue(value=selection_q, type=RadialUnitType.Q_NM)

    # Load CCF data
    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback) as catcher:
        acc = NpxFile.load(ccf_file)
        for key in ['radial', 'delta', 'ccf']:
            if key not in acc.data.keys():
                cli.print(f'Error: Missing key {key!r} in CCF file {ccf_file}', style='red')
                raise typer.Abort()
    if catcher.raised:
        cli.print(f'Error: Failed to load CCF file {ccf_file}', style='red')
        raise typer.Abort()

    # Load azimuthal integration data
    has_ai2 = True if ai2_file is not None else False
    if has_ai2:
        with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback) as catcher:
            ai2 = NpxFile.load(ai2_file)
            for key in ['radial', 'azimuthal', 'intensities']:
                if key not in ai2.data.keys():
                    cli.print(f'Error: Missing key {key!r} in 2D azimuthal integration file {ccf_file}',
                              style='red')
                    raise typer.Abort()
        if catcher.raised:
            cli.print(f'Error: Failed to load azimuthal integration file {ai2_file}', style='red')
            raise typer.Abort()

    # Get geometry from CCF file
    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback) as catcher:
        geometry = get_geometry_from_header(acc.header)
        if verbose > 2:
            cli.print('Geometry read from CCF file:', style='cyan')
            cli.pprint(dict(geometry.get_config()))
    if catcher.raised:
        cli.print(f'Error: Failed to parse geometry information from {ccf_file}', style='red')
        raise typer.Abort()

    # Get the radial units of the CCF
    r = get_radial_units_from_header(acc.header)
    if r is None:
        cli.print('Error: Failed to get radial units from CCF file header', style='red')
        raise typer.Abort()
    cli.print(f'Radial units in CCF file: {r.name.latex} ({r.unit.latex})')

    # Convert selection units to the units of CCF if necessary
    wavelength = geometry.wavelength / 1e-10
    if r.type != selection.type:
        with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback) as catcher:
            n0, v0, u0 = selection.name.latex, selection.value, selection.units.latex
            selection = selection.to(units=r, wavelength=wavelength)
            n1, v1, u1 = selection.name.latex, selection.value, selection.units.latex

            u0 = u0 if u0 == '\u00b0' else f' {u0}'
            u1 = u1 if u1 == '\u00b0' else f' {u1}'
            cli.print(f'Converted selection from {n0}={v0:.4f}{u0} '
                      f'to {n1}={v1:.4f}{u1} using \u03bb={wavelength:.4f} \u212b')

    # Check if selection is within the radial range of the CCF
    radial_min, radial_max = acc.data['radial'].min(), acc.data['radial'].max()
    if selection.value < radial_min or selection.value > radial_max:
        cli.print(f'Error: Selection is outside the radial range of the CCF: {[radial_min, radial_max]}', style='red')
        raise typer.Abort()

    # Get the radial index of the selection
    ccf_i = np.argmin(np.abs(acc.data['radial'] - selection.value))

    # Calculate the FFT of the CCF
    if verbose:
        cli.print(f'Calculating FFT... ')
    with Timer(silent=verbose<1, echo_func=cli.print):
        ccf = acc.data['ccf']
        ccf_delta = ccf[:, ccf_i]
        if np.any(np.isnan(ccf_delta)):
            if fill_value is not None:
                cli.print(f'Replacing missing values in CCF with {fill_value}', style='yellow')
                ccf_delta = np.nan_to_num(ccf_delta, nan=fill_value)
            else:
                cli.print('Warning: NaN values found in CCF data. FFT will be empty. '
                          'Use --fill to replace the NaNs.', style='yellow')
        fc = np.fft.fft(ccf_delta) / len(ccf_delta)
        fc = np.abs(fc)

    # Calculate selection in 2theta, q and d
    tth = selection.to(RadialUnitType.TWOTHETA_DEG, wavelength=wavelength)
    q = selection.to(RadialUnitType.Q_NM, wavelength=wavelength)
    d = selection.to(RadialUnitType.D_A, wavelength=wavelength)
    subtitle = (f'{tth.name.latex}={tth.value:.4f}{tth.units.latex} | '
                f'{q.name.latex}={q.value:.4f} {q.units.latex} | '
                f'{d.name.latex}={d.value:.2f} {d.units.latex}')

    # Prepare plots
    fig = plt.figure('XCCA overview', figsize=(16 / 1.2, 9 / 1.2))
    fig.suptitle(f'{ccf_file.name}\n{subtitle}')
    gs = fig.add_gridspec(2, 3, wspace=0.2,)
    ax0 = fig.add_subplot(gs[0, 0])  # CCF 2D
    ax1 = fig.add_subplot(gs[1, 0])  # Azimuthal integration 2D
    ax2 = fig.add_subplot(gs[0, 1])  # CCF 1D
    # ax2 = fig.add_subplot(gs[0, 1], projection='polar')  # CCF 1D
    ax3 = fig.add_subplot(gs[1, 1])  # Azimuthal integration 1D
    # ax3 = fig.add_subplot(gs[1, 1], projection='polar')  # Azimuthal integration 1D
    ax4 = fig.add_subplot(gs[:, 2])  # FFT

    for ax in [ax0, ax1]:
        ax.tick_params(direction='in', color='white', bottom=True, top=True, left=True,
                        right=True)
    for ax in [ax2, ax3, ax4]:
        ax.tick_params(direction='in', bottom=True, left=True)

    ccf, radial, delta = acc.data['ccf'], acc.data['radial'], acc.data['delta']

    # CCF
    vmin, vmax = np.nanmin(ccf), np.nanmax(ccf)
    if normalize:
        mean = np.nanmean(ccf)
        std = np.nanstd(ccf)
        if zmin is not None and verbose:
            cli.print(f'Skipping zmin normalization: already set', style='yellow')
        else:
            vmin = np.max([mean - nstd * std, vmin])
        if zmax is not None and verbose:
            cli.print(f'Skipping zmax normalization: already set', style='yellow')
        else:
            vmax = np.min([mean + nstd * std, vmax])
        if verbose:
            cli.print(
                f'Normalization: mean={mean:.4e}, std={std:.4e}, zmin={vmin:.4e}, zmax={vmax:.4e}',
                style='cyan')
    else:
        if zmin is not None:
            vmin = zmin
        if zmax is not None:
            vmax = zmax

    if zscale == ZScale.LINEAR:
        norm = partial(Normalize, clip=False)
    elif zscale == ZScale.LOG:
        norm = partial(SymLogNorm, clip=False, linthresh=0.05)

    ax0.imshow(ccf, origin='lower', aspect='auto', interpolation='nearest', cmap='Spectral',
               norm=norm(vmin=vmin, vmax=vmax),
               extent=(radial.min(), radial.max(), delta.min(), delta.max()))
    ax0.vlines(selection.value, delta.min(), delta.max(), 'r', '--')
    ax0.set_title('2D Cross-correlation function')
    ax0.set_ylabel(f'\u0394 (\u00b0)')
    ax0.set_xlabel(f'{selection.name.latex} ({selection.units.latex})')

    ax2.plot(delta, ccf[:, ccf_i], color='xkcd:light brown')
    # ax2.plot(delta * np.pi / 180., ccf[:, ccf_i], color='xkcd:light brown')
    ax2.set_title('1D Cross-correlation function')
    ax2.set_ylabel('Cross-correlation function')
    ax2.set_xlabel('\u03c7 (\u00b0)')

    if has_ai2:
        intensities, radial, azimuthal = ai2.data['intensities'], ai2.data['radial'], ai2.data['azimuthal']

        # AI2
        vmin, vmax = np.nanmin(intensities), np.nanmax(intensities)
        if normalize:
            mean = np.nanmean(intensities)
            std = np.nanstd(intensities)
            if zmin is not None and verbose:
                cli.print(f'Skipping zmin normalization: already set', style='yellow')
            else:
                vmin = np.max([mean - nstd * std, vmin])
            if zmax is not None and verbose:
                cli.print(f'Skipping zmax normalization: already set', style='yellow')
            else:
                vmax = np.min([mean + nstd * std, vmax])
            if verbose:
                cli.print(
                    f'Normalization: mean={mean:.4e}, std={std:.4e}, zmin={vmin:.4e}, zmax={vmax:.4e}',
                    style='cyan')
        else:
            if zmin is not None:
                vmin = zmin
            if zmax is not None:
                vmax = zmax

        if zscale == ZScale.LINEAR:
            norm = partial(Normalize, clip=False)
        elif zscale == ZScale.LOG:
            norm = partial(SymLogNorm, clip=False, linthresh=0.05)

        ax1.imshow(intensities, origin='lower', aspect='auto', interpolation='nearest', cmap='magma',
                   norm=norm(vmin=vmin, vmax=vmax),
                   extent=(radial.min(), radial.max(), azimuthal.min(), azimuthal.max()))
        ax1.vlines(selection.value, azimuthal.min(), azimuthal.max(), 'r', '--')
        ax1.set_title('2D Azimuthal integration')
        ax1.set_ylabel('\u03c7 (\u00b0)')
        ax1.set_xlabel(f'{selection.name.latex} ({selection.units.latex})')

        ax3.plot(azimuthal, intensities[:, ccf_i], color='xkcd:crimson')
        # ax3.plot(azimuthal * np.pi / 180., intensities[:, ccf_i], color='xkcd:crimson')
        # ax3.set_ylim(intensities[:, ccf_i].min(), intensities[:, ccf_i].max())
        ax3.set_title('1D Azimuthal integration')
        ax3.set_ylabel('Intensity')
        ax3.set_xlabel('\u03c7 (\u00b0)')

    ax4.bar(range(1, no_coeffs + 1), fc[1:no_coeffs+1])
    ax4.set_title('$\u2131\{\mathrm{CCF}(\u0394)\}$')
    ax4.set_ylabel('Distribution')
    ax4.set_xlabel('Fourier coefficient')

    plt.show()
