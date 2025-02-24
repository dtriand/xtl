from pathlib import Path

from pyFAI.detectors import ALL_DETECTORS, Detector
from pyFAI.geometry import Geometry
from tabulate import tabulate
import typer

from xtl.cli.cliio import CliIO


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


@app.command('mask', help='Create a detector mask (interactive)')
def cli_difftegrate_mask():
    cli = CliIO()
    cli.echo('NotImplementedError', level='error')
    raise typer.Abort()


@app.command('1d', help='Perform 1D integration')
def cli_difftegrate_1d():
    cli = CliIO()
    cli.echo('NotImplementedError', level='error')
    raise typer.Abort()


@app.command('2d', help='Perform 2D integration')
def cli_difftegrate_2d():
    cli = CliIO()
    cli.echo('NotImplementedError', level='error')
    raise typer.Abort()


