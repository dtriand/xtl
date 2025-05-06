from pathlib import Path

from pyFAI.detectors import ALL_DETECTORS, Detector
from pyFAI.geometry import Geometry
import typer

from xtl.cli.cliio import Console


app = typer.Typer()


def get_detector_info(detector: Detector) -> dict:
    detector_info = {}
    detector_info['Detector name'] = detector.name
    if isinstance(detector.MANUFACTURER, list):
        detector_info['Manufacturer'] = ' '.join(detector.MANUFACTURER)
    else:
        detector_info['Manufacturer'] = detector.MANUFACTURER
    if detector.pixel1 is not None:
        pixel_size = f'{detector.pixel1 * 1e6:.2f}\u00d7{detector.pixel2 * 1e6:.2f}'
    else:
        pixel_size = 'N/A'
    detector_info['Pixel size (h\u00d7v) \[\u03bcm]'] = pixel_size
    if detector.shape is not None:
        dimensions = f'{detector.shape[1]}\u00d7{detector.shape[0]}'
    else:
        dimensions = 'N/A'
    detector_info['Dimensions (h\u00d7v) \[px]'] = dimensions
    return detector_info


def get_detectors_list() -> dict:
    detectors = {}
    for alias, detector in ALL_DETECTORS.items():
        detector = detector()
        if detector.name not in detectors:
            detectors[detector.name] = {
                'detector': detector,
                'aliases': [alias]
            }
        else:
            detectors[detector.name]['aliases'].append(alias)

    for name in detectors.keys():
        detector = detectors[name]['detector']
        info = get_detector_info(detector)
        info['Aliases'] = ', '.join(detectors[name]['aliases'])
        detectors[name]['info'] = info
    return {k: v['info'] for k, v in detectors.items()}


def get_geometry_info(geometry: Geometry) -> dict:
    return {f'{k}': f'{v}' for k, v in geometry.get_config().items() if k not in ['poni_version', 'detector_config']}


@app.command('geometry', help='Create a .poni file for describing detector geometry (interactive)')
def cli_diffraction_geometry():
    cli = Console()

    # Determine detector
    detector = None
    while detector is None:
        answer = typer.prompt('Enter detector name (? for list)').lower()
        if answer in ['detector', 'custom']:
            cli.print('Initializing custom detector...')
            pixel1 = typer.prompt('Enter pixel size (horizontal) [pixel1 in \u03bcm]', default=50., type=float)
            pixel2 = typer.prompt('Enter pixel size (vertical) [pixel2 in \u03bcm]', default=50., type=float)
            shape1 = typer.prompt('Enter number of pixels (horizontal) [shape[1] in px]', default=1024, type=int)
            shape2 = typer.prompt('Enter number of pixels (vertical) [shape[0] in px', default=1024, type=int)
            detector = Detector(pixel1=pixel1/1e6, pixel2=pixel2/1e6, max_shape=(shape2, shape1))
        elif answer in ALL_DETECTORS:
            detector: Detector = ALL_DETECTORS[answer]()
        else:
            detector_list = get_detectors_list()
            headers = list(list(detector_list.values())[0].keys())
            cli.print_table([d.values() for d in detector_list.values()], headers=headers,
                            table_kwargs={'box': None, 'title': 'Available detectors'})
            if answer not in ['?', '']:
                cli.print(f'Unknown detector alias: {answer!r}', style='red')
                cli.print('Choose a detector alias from the list above.')
            continue

        det_info = get_detector_info(detector)
        cli.print_table(det_info.items(), headers=['', ''],
                        table_kwargs={'show_header': False, 'box': None})
        ok = cli.confirm('Is this the correct detector?', default=True)
        if not ok:
            detector = None

    # Build Geometry object
    geometry = None
    while geometry is None:
        poni2 = typer.prompt('Enter point of normal incidence along horizontal (x) axis [poni2 in px]', default=0., type=float)
        poni1 = typer.prompt('Enter point of normal incidence along vertical (y) axis [poni1 in px]', default=0., type=float)
        dist = typer.prompt('Enter sample-detector distance [dist in mm]', default=100., type=float)
        rot2 = typer.prompt('Enter detector rotation around horizontal (x) axis [rot2 in \u00b0]', default=0., type=float)
        rot1 = typer.prompt('Enter detector rotation around vertical (y) axis [rot1 in \u00b0]', default=0., type=float)
        rot3 = typer.prompt('Enter detector rotation around beam (z) axis [rot3 in \u00b0]', default=0., type=float)
        wavelength = typer.prompt('Enter wavelength [wave in \u212b]', default=1., type=float)

        geometry = Geometry(
            detector=detector,
            poni1 = poni1 * detector.pixel1,  # convert to meters
            poni2 = poni2 * detector.pixel2,  # convert to meters
            dist = dist / 1000.,  # convert to meters
            rot1 = rot1,  # in degrees
            rot2 = rot2,  # in degrees
            rot3 = rot3,  # in degrees
            wavelength = wavelength * 1e-10  # convert to meters
        )

        geometry_info = get_geometry_info(geometry)
        cli.print_table(geometry_info.items(), headers=['', ''],
                        table_kwargs={'show_header': False, 'box': None})
        ok = cli.confirm('Is this the correct geometry?', default=True)
        if not ok:
            geometry = None

    # Save Geometry object to a .poni file
    cwd = Path.cwd().expanduser().resolve()
    file = None
    while file is None:
        answer = typer.prompt('Enter filename for .poni file')
        file = Path(answer)
        if file.parent == '.':
            file = cwd / file
        if file.suffix != '.poni':
            file = file.with_suffix('.poni')
        file = file.resolve()
        if file.exists():
            ok = cli.confirm(f'File {file} already exists. Overwrite?', default=False)
            if not ok:
                file = None
                continue
            file.unlink(missing_ok=True)
        geometry.save(file)
        cli.print(f'Geometry saved to {file}', style='green')
