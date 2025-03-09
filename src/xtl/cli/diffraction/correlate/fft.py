from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import typer

from xtl.cli.cliio import Console, epilog
from xtl.cli.utils import Timer
from xtl.cli.diffraction.cli_utils import (IntegrationRadialUnits, get_geometry_from_header,
                                           get_radial_units_from_header, units_repr)
from xtl.diffraction.images.correlators import AzimuthalCrossCorrelatorQQ_1
from xtl.exceptions.utils import Catcher
from xtl.files.npx import NpxFile

import warnings


app = typer.Typer()


@app.command('fft', help='Calculate the distribution of Fourier components', epilog=epilog)
def cli_diffraction_correlate_fft(
        ccf_file: Path = typer.Argument(..., metavar='ccf.npx', help='CCF file to analyze'),
        ai2_file: Path = typer.Argument(None, metavar='ai2.npx', help='Azimuthal integration file'),
        # Selection parameters
        selection_2theta: float = typer.Option(None, '-tt', '--2theta', help='2\u03b8 angle to inspect (in \u00b0)',
                                               rich_help_panel='Selection parameters'),
        selection_q: float = typer.Option(None, '-q', '--q', help='Q value to inspect (in nm\u207B\u00B9)',
                                          rich_help_panel='Selection parameters'),
        # FFT parameters
        no_coeffs: int = typer.Option(24, '-n', '--no-coeffs', help='Number of Fourier components to display',
                                      rich_help_panel='FFT parameters'),
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
    selection_units = IntegrationRadialUnits.Q_NM if selection_q is not None else IntegrationRadialUnits.TWOTHETA_DEG
    selection = selection_q if selection_q is not None else selection_2theta

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
    radial_units = get_radial_units_from_header(acc.header)
    if radial_units is None:
        cli.print('Error: Failed to get radial units from CCF file header', style='red')
        raise typer.Abort()
    u = units_repr(radial_units)
    cli.print(f'Radial units in CCF file: {u[0]} ({u[1]})')

    # Convert selection units to the units of CCF if necessary
    if radial_units != selection_units:
        previous_selection = selection
        if radial_units == IntegrationRadialUnits.Q_NM:  # Convert 2theta to q
            selection = 4 * np.pi / (geometry.wavelength / 1e-10) * np.sin(np.radians(selection / 2))
            selection *= 10  # Convert 1/A to nm^-1
        elif radial_units == IntegrationRadialUnits.TWOTHETA_DEG:  # Convert q to 2theta
            selection /= 10  # Convert nm^-1 to 1/A
            selection = 2 * np.degrees(np.arcsin(selection * (geometry.wavelength / 1e-10) / (4 * np.pi)))
        else:
            cli.print(f'Error: Invalid radial units {radial_units!r}', style='red')
            raise typer.Abort()
        u0 = units_repr(selection_units)
        u1 = units_repr(radial_units)
        cli.print(f'Converted selection from {u0[0]}={previous_selection:.4f} ({u0[1]}) '
                  f'to {u1[0]}={selection:.4f} ({u1[1]}) using \u03bb={geometry.wavelength / 1e-10:.4f} (\u212b)')

    # Check if selection is within the radial range of the CCF
    radial_min, radial_max = acc.data['radial'].min(), acc.data['radial'].max()
    if selection < radial_min or selection > radial_max:
        cli.print(f'Error: Selection is outside the radial range of the CCF: {[radial_min, radial_max]}', style='red')
        raise typer.Abort()

    # Get the radial index of the selection
    ccf_i = np.argmin(np.abs(acc.data['radial'] - selection))

    # Calculate the FFT of the CCF
    if verbose:
        cli.print(f'Calculating FFT(CCF)... ')
    with Timer(silent=verbose<=1, echo_func=cli.print):
        ccf = acc.data['ccf']
        ccf_delta = ccf[:, ccf_i]
        fc = np.fft.fft(ccf_delta) / len(ccf_delta)
        fc = np.abs(fc)

    # Prepare plots
    fig = plt.figure('XCCA overview', figsize=(16 / 1.2, 9 / 1.2))
    fig.suptitle(f'{ccf_file.name} {u[0]}={selection:.4f} {u[1]}')
    gs = fig.add_gridspec(2, 3, wspace=0.2,)
    ax0 = fig.add_subplot(gs[0, 0])  # CCF 2D
    ax1 = fig.add_subplot(gs[1, 0])  # Azimuthal integration 2D
    ax2 = fig.add_subplot(gs[0, 1])  # CCF 1D
    ax3 = fig.add_subplot(gs[1, 1])  # Azimuthal integration 1D
    ax4 = fig.add_subplot(gs[:, 2])  # FFT

    ccf, radial, delta = acc.data['ccf'], acc.data['radial'], acc.data['delta']
    ax0.imshow(ccf, origin='lower', aspect='auto', interpolation='nearest', cmap='Spectral',
               #norm=norm(vmin=zmin, vmax=zmax),
               extent=(radial.min(), radial.max(), delta.min(), delta.max()))
    ax0.vlines(selection, delta.min(), delta.max(), 'r', '--')
    ax0.set_title('2D Cross-correlation function')
    ax0.set_ylabel(f'Angular offset / \u0394 ({u[1]})')
    ax0.set_xlabel(f'Radial angle / {u[0]} ({u[1]})')

    ax2.plot(delta, ccf[:, ccf_i])
    ax2.set_title('1D Cross-correlation function')
    ax2.set_ylabel('Cross-correlation function')
    ax2.set_xlabel(f'Angular offset / \u0394 ({u[1]})')

    if has_ai2:
        intensities, radial, azimuthal = ai2.data['intensities'], ai2.data['radial'], ai2.data['azimuthal']
        ax1.imshow(intensities, origin='lower', aspect='auto', interpolation='nearest', cmap='magma',
                   #norm=norm(vmin=zmin, vmax=zmax),
                   extent=(radial.min(), radial.max(), azimuthal.min(), azimuthal.max()))
        ax1.vlines(selection, azimuthal.min(), azimuthal.max(), 'r', '--')
        ax1.set_title('2D Azimuthal integration')
        ax1.set_ylabel(f'Azimuthal angle / \u03c7 {u[1]}')
        ax1.set_xlabel(f'Radial angle / {u[0]} ({u[1]})')

        ax3.plot(azimuthal, intensities[:, ccf_i])
        ax3.set_title('1D Azimuthal integration')
        ax3.set_ylabel('Intensity')
        ax3.set_xlabel(f'Azimuthal angle / \u03c7 ({u[1]})')

    ax4.bar(range(1, no_coeffs + 1), fc[1:no_coeffs+1])
    ax4.set_title('FFT(CCF)')
    ax4.set_ylabel('Distribution')
    ax4.set_xlabel('Fourier coefficient')

    plt.show()

    # ttheta = img.ai2.results.radial
    # chi = img.ai2.results.azimuthal
    # intensities = img.ai2.results.intensity
    # tth = inspect_2theta
    # itth = np.argmin(np.abs(ttheta - tth))
    #
    # # Convert 2theta to d-spacing, q-spacing and radial distance in pixels
    # d = (img.geometry.wavelength / 1e-10) / (2 * np.sin(np.radians(ttheta[itth] / 2)))  # in angstroms
    # q = 4 * np.pi / (d / 10)  # in reciprocal nanometers
    # r = img.geometry.dist * np.tan(np.radians(ttheta[itth])) / img.geometry.pixel1  # 2theta radius in pixel coordinates
    #
    # # Plots
    # fig2 = plt.figure('Inspection', figsize=(5, 7))
    # gs = fig.add_gridspec(3, 1)
    # ax5 = fig2.add_subplot(gs[0, 0])  # Normalized CCF
    # ax6 = fig2.add_subplot(gs[1, 0])  # Normalized intensity across azimuthal angle
    # ax7 = fig2.add_subplot(gs[2, 0])  # Fourier coefficients histogram
    #
    # # CCF
    # ccf_delta_chi = ccf.T[itth]
    # ax5.plot(chi, ccf_delta_chi)
    # ax5.set_xlabel('Angular offset / \u0394 (\u00b0)')
    # ax5.set_ylabel('Cross-correlation function')
    #
    # # Intensity
    # I_chi = intensities.T[itth]
    # ax6.plot(chi, I_chi, color='tab:orange')
    # ax6.set_xlabel('Azimuthal angle / \u03c7 (\u00b0)')
    # ax6.set_ylabel('Intensity (A.U.)')
    #
    # # CCF fourier components
    # no_coeffs = 24
    # fc = np.fft.fft(ccf_delta_chi) / len(ccf_delta_chi)
    # fc = np.abs(fc)
    # ax7.bar(range(1, no_coeffs + 1), fc[1:no_coeffs+1])
    # ax7.set_xlabel('Fourier coefficient')
    # ax7.set_ylabel('Distribution')
    #
    # # Highlight the selected 2theta segment in the previous plots
    # ax0.add_artist(Circle(img.beam_center, r, fill=False, edgecolor='r', linestyle='--'))
    # ax1.vlines(ttheta[itth], chi.min(), chi.max(), 'r', '--')
    # ax2.vlines(ttheta[itth], *ax2.get_ylim(), 'r', '--')
    # ax3.vlines(ttheta[itth], chi.min(), chi.max(), 'r', '--')
    # ax4.vlines(ttheta[itth], ccf_mean.min(), ccf_mean.max(), 'r', '--')
    #
    # fig2.suptitle(f'2\u03b8={ttheta[itth]:.4f}\u00b0 / q={q:.4f} nm$^{{-1}}$ / d={d:.2f} \u212b')