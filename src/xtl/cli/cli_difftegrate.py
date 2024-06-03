from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from pyFAI.detectors import ALL_DETECTORS, Detector
from pyFAI.geometry import Geometry
from tabulate import tabulate
import typer

from xtl.cli.cliio import CliIO
from xtl.diffraction.images.images import Image
from xtl.diffraction.images.integrators import AzimuthalIntegrator1D, AzimuthalIntegrator2D
from xtl.math import si_units


app = typer.Typer(name='difftegrate', help='Perform azimuthal integrations of x-ray data', add_completion=False)


def get_detectors_list():
    detectors_list = []
    detectors_dict = {}

    for detname, det in ALL_DETECTORS.items():
        detector = det()
        if detector.name not in detectors_dict:
            detectors_dict[detector.name] = {
                'detector': detector,
                'alias': [detname]
            }
        else:
            detectors_dict[detector.name]['alias'].append(detname)

    for detname in detectors_dict.keys():
        detector = detectors_dict[detname]['detector']
        aliases = ', '.join(detectors_dict[detname]['alias'])
        manufacturer = ' '.join(detector.MANUFACTURER) if isinstance(detector.MANUFACTURER, list) \
            else detector.MANUFACTURER
        pixel_size = '\u00d7'.join([f'{pix*1e6:.2f}' for pix in (detector.pixel2, detector.pixel1) if pix is not None])
        detectors_list.append([manufacturer, detname, pixel_size, aliases])
    return detectors_list


def get_detector_info(detector: Detector):
    detector_info = []
    detector_info.append(['Detector name', detector.name])
    manufacturer = ' '.join(detector.MANUFACTURER) if isinstance(detector.MANUFACTURER, list) else detector.MANUFACTURER
    detector_info.append(['Manufacturer', manufacturer])
    if detector.pixel1 is not None:
        pixel_size = '\u00d7'.join([f'{pix * 1e6:.2f}' for pix in (detector.pixel2, detector.pixel1)]) + ' \u03bcm'
    else:
        pixel_size = None
    detector_info.append(['Pixel size (h\u00d7v)', pixel_size])
    if detector.shape is not None:
        dimensions = '\u00d7'.join(str(dim) for dim in detector.shape[::-1]) + ' pixels'
    else:
        dimensions = None
    detector_info.append(['Dimensions (h\u00d7v)', dimensions])
    return detector_info


@app.command('geometry', help='Create a .poni file for describing detector geometry (interactive)')
def cli_difftegrate_geometry():
    cli = CliIO()
    cwd = Path.cwd()
    cli.echo(f'Current working directory: {cwd}')

    # Determine detector
    detector = None
    while detector is None:
        answer = typer.prompt('Detector name')
        if answer in ALL_DETECTORS:
            detector = ALL_DETECTORS[answer]()
            cli.echo(tabulate(get_detector_info(detector), tablefmt='simple', colalign=['right', 'left']))
            ok = typer.confirm('Is this the correct detector?', default=True)
            if not ok:
                detector = None
        else:
            if answer == '?':
                cli.echo('Available detectors (from pyFAI): ')
            else:
                cli.echo(f'Unknown detector \'{answer}\'. Choose one from: ')
            cli.echo(tabulate(get_detectors_list(),
                              headers=['Manufacturer', 'Detector name', 'Pixel size (h\u00d7v \u03bcm)', 'Identifiers'],
                              colalign=['right', 'left', 'left', 'left']))
            cli.echo('Choose a detector identifier from the list above')

    # Determine the rest of the poni parameters (in useful units)
    params = ['poni1', 'poni2', 'dist', 'rot1', 'rot2', 'rot3', 'wavelength']
    values = [None, None, None, None, None, None, None]
    units = ['pixel', 'pixel', 'mm', '\u00b0', '\u00b0', '\u00b0', '\u212b']
    descriptions = [
        'Point Of Normal Incidence along vertical (y) axis',
        'Point Of Normal Incidence along horizontal (x) axis',
        'Sample-to-detector distance',
        'Detector rotation along vertical (y) axis',
        'Detector rotation along horizontal (x) axis',
        'Detector rotation along beam (z) axis',
        'Radiation wavelength'
    ]
    for i, (param, value, unit, description) in enumerate(zip(params, values, units, descriptions)):
        while value is None:
            answer = typer.prompt(f'{description} [{param}:{unit}]')
            try:
                value = float(answer)
                values[i] = value
            except ValueError:
                cli.echo(f'Invalid type {answer.__class__.__name__}. Must be a number!', level='warning')

    # Convert values to pyfai units
    poni1 = values[0] * detector.pixel1  # in meters
    poni2 = values[1] * detector.pixel2  # in meters
    dist = values[2] * 0.001  # in meters
    rot1 = values[3]
    rot2 = values[4]
    rot3 = values[5]
    wavelength = values[6] * 1e-10  # in meters

    # Build Geometry object
    poni = Geometry(detector=detector, poni1=poni1, poni2=poni2, dist=dist, rot1=rot1, rot2=rot2, rot3=rot3,
                    wavelength=wavelength)
    cli.echo('The following .poni file has been created:')
    cli.echo('\n'.join(f'  {k}: {v}' for k, v in poni.get_config().items()))

    # Save Geometry object to a .poni file
    file = None
    while file is None:
        answer = typer.prompt('Provide filename to save .poni file')
        file = Path(answer)
        if file.parent == '.':
            file = cwd / answer
        if file.suffix != '.poni':
            file = file.with_suffix('.poni')
        if file.exists():
            ok = typer.prompt(f'File {file} already exists. Would you like to overwrite it?', default=True)
            if not ok:
                file = None
                continue
            file.unlink(missing_ok=True)
        poni.save(file)
        cli.echo(f'Saved file {file.resolve()}')
    return


@app.command('mask', help='Create a detector mask (interactive)')
def cli_difftegrate_mask():
    cli = CliIO()
    cli.echo('NotImplementedError', level='error')
    raise typer.Abort()


@app.command('1d', help='Perform 1D integration')
def cli_difftegrate_1d(file: Path = typer.Option(None, '-i', '--input', prompt='Input image file',
                                                 help='Image file to integrate'),
                       poni: Path = typer.Option(None, '-g', '--geometry', prompt='Geometry .poni file',
                                                 help='Geometry .poni file'),
                       output: Path = typer.Option(None, '-o', '--output', prompt='Output file',
                                                   help='Save integration results to file'),
                       frame: int = typer.Option(0, '-f', '--frame', help='Starting image frame for integration'),
                       no_frames: int = typer.Option(1, '-n', '--no_frames', help='Number of frames to integrate'),
                       average: bool = typer.Option(False, '-a', '--average', help='Average integration results'),
                       plot: bool = typer.Option(False, '-p', '--plot', help='Plot integration results')
                       ):
    # Check if files exist
    cli = CliIO()
    for f in (file, poni):
        if not f.exists():
            cli.echo(f'File {f} does not exist', level='error')
            raise typer.Abort()

    # Check frame numbers
    if frame < 0:
        cli.echo('Frame number must be a positive integer!', level='error')
        raise typer.Abort()
    if no_frames < 1:
        cli.echo('Number of frames must be at least 1!', level='error')
        raise typer.Abort()

    # Load data
    img = Image()
    img.open(file=file, frame=frame, is_eager=True)
    cli.echo(f'Loaded frame #{img.frame} from {img.file}')
    img.load_geometry(poni)
    cli.echo(f'Loaded geometry from {poni}')

    img.mask.mask_detector('eiger_4m')
    img.mask.mask_intensity_greater_than(1e9)

    # Initialize integrator
    ai1 = AzimuthalIntegrator1D(img)
    ai1.initialize(error_model='poisson')
    npt_rad = ai1.points_radial
    cli.echo(f'Performing 1D azimuthal integration on {npt_rad} 2\u03b8 points... ')

    results = []
    for i in range(frame, frame + no_frames):
        # Perform integration and store results
        t1 = datetime.now()
        results.append(ai1.integrate())
        t2 = datetime.now()
        t_delta = (t2 - t1).total_seconds()
        cli.echo(f'Integration for frame #{i:05} completed in {t_delta:.3f} sec '
                 f'({si_units(t_delta / npt_rad, suffix="sec/point", digits=3)})')

        # Prepare plot
        if plot:
            errors = True if no_frames == 1 else False
            ax, fig = ai1.plot(label=f'#{i:05}', errors=errors)

        # Save integration results to .xye file
        fname = output.parent / f'{output.stem}_f{i:05}.xye'
        ai1.save(fname, overwrite=True, header=False)
        cli.echo(f'Saved results to file {fname}')

        # Load next frame
        try:
            if i == (no_frames - 1):
                continue
            img.next_frame()
        except Exception:
            cli.echo('Run out of frames. Stopping integration.')

    if average:
        intensities = results[0].intensity
        for result in results[1:]:
            intensities += result.intensity
        intensities /= len(results)
        if plot:
            ax.plot(result.radial, intensities, '--', label='Average')

    if plot:
        ax.legend(title='Frame no.')
        plt.show()


@app.command('2d', help='Perform 2D integration')
def cli_difftegrate_2d(file: Path = typer.Option(None, '-i', '--input', prompt='Input image file',
                                                 help='Image file to integrate'),
                       poni: Path = typer.Option(None, '-g', '--geometry', prompt='Geometry .poni file',
                                                 help='Geometry .poni file'),
                       output: Path = typer.Option(None, '-o', '--output', prompt='Output file',
                                                   help='Save integration results to file'),
                       frame: int = typer.Option(0, '-f', '--frame', help='Starting image frame for integration')):
    # Check if files exist
    cli = CliIO()
    for f in (file, poni):
        if not f.exists():
            cli.echo(f'File {f} does not exist', level='error')
            raise typer.Abort()

    img = Image()
    img.open(file=file, frame=frame, is_eager=True)
    cli.echo(f'Loaded frame {img.frame} from {img.file}')
    img.load_geometry(poni)
    cli.echo(f'Loaded geometry from {poni}')

    img.mask.mask_detector('eiger_4m')
    img.mask.mask_intensity_greater_than(1e9)

    ai2 = AzimuthalIntegrator2D(img)
    ai2.initialize(error_model='poisson')
    npt_rad, npt_azim = ai2.points_radial, ai2.points_azimuthal
    cli.echo(f'Performing 2D azimuthal integration on a {npt_azim}\u00d7{npt_rad} grid (\u03c7\u00d72\u03b8)')

    t1 = datetime.now()
    ai2.integrate()
    t2 = datetime.now()
    t_delta = (t2 - t1).total_seconds()
    cli.echo(f'Integration complete! Time elapsed: {t_delta:.4f} sec '
             f'({si_units(t_delta / (npt_rad * npt_azim), suffix="sec/point", digits=3)})')
    results = ai2.results
    # plt.imshow(results.intensity, origin='lower', aspect='auto', extent=(results.radial.min(), results.radial.max(),
    #                                                                      results.azimuthal.min(), results.azimuthal.max()))
    # plt.show()
    f = (Path('D:/') / 'code' / 'xtl_scripts' / 'test.npx').resolve()
    ai2.save(f, overwrite=True, header=True)
    cli.echo(f'Saved results to file {f}')
    ai2.plot(zscale='log', overlay_mask=True)
    img.initialize_azimuthal_integrator(1)
    plt.show()


