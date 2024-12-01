import copy
from pathlib import Path
from typing import Callable

import typer

from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig


def str_or_none(s):
    if s is None:
        return None
    return str(s)


def get_param_type(t: tuple):
    a, b = t
    if not b:
        return a.__name__
    return f'{a.__name__}[{b.__name__}]'


def get_attributes_config() -> list[list[str]]:
    attributes = []
    c = AutoPROCConfig()
    for key, field in c.__dataclass_fields__.items():
        if field.metadata.get('param_type') in ['__internal']:
            continue
        if key.startswith('_'):  # hide attributes like _macros and _args
            continue
        if field.metadata.get('cli_hidden', False):  # hide tagged attributes
            continue
        alias = field.metadata.get('alias', None)
        if alias and alias.startswith('_'):
            alias = None
        description = field.metadata.get('desc', None)
        attribute_type = get_param_type(c._derive_type(field))
        attributes.append([key, alias, attribute_type, description])
    return attributes


def get_attributes_dataset() -> list[list[str]]:
    attributes = [
        ['raw_data_dir', 'Path', 'Path to the raw data directory'],
        ['dataset_dir', 'str', 'Subdirectory within the raw data directory'],
        ['dataset_name', 'str', 'Name of the dataset excluding image number'],
        ['first_image', 'Path', 'Full path to the first image file'],
        ['processed_data_dir', 'Path', 'Path to the processed data directory'],
        ['output_dir', 'str', 'Subdirectory within the processed data directory'],
        ['output_subdir', 'str', 'Subdirectory within the output directory']
    ]
    return attributes


def parse_csv2(csv_file: Path, extra_headers: list[str] = None):
    parsable_attrs_dataset = [attr[0] for attr in get_attributes_dataset()]
    parsable_attrs_config = [attr[0] for attr in get_attributes_config()]

    csv_dict = {
        'dataset': {key: [] for key in parsable_attrs_dataset},
        'config': {key: [] for key in parsable_attrs_config},
        'extra': {},
        'index': {key: 'dataset' for key in parsable_attrs_dataset} |
                     {key: 'config' for key in parsable_attrs_config},
        'headers': []
    }

    if extra_headers:
        csv_dict['extra'] = {key: [] for key in extra_headers}
        csv_dict['index'].update({key: 'extra' for key in extra_headers})

    indices = {key: None for key in csv_dict['index'].keys()}

    headers = csv_file.read_text().splitlines()[0].replace('#', '').replace(' ', '').split(',')
    for i, header in enumerate(headers):
        if header in indices.keys():
            indices[header] = i

    for line in csv_file.read_text().splitlines()[1:]:
        # Skip commented lines
        if line.startswith('#'):
            continue

        # Split each line to a list of values
        values = line.split(',')

        # Iterate over the keys and append the values to the corresponding list
        for key, group in csv_dict['index'].items():
            # Skip keys that have not been found in the csv header
            if indices[key] is None:
                csv_dict[group][key].append(None)
            else:
                # Get the value for the key by its index
                v = values[indices[key]]
                if not v:
                    v = None
                csv_dict[group][key].append(v)

    # Set the keys that have been found in the csv header to the headers list
    csv_dict['headers'] = [key for key, value in indices.items() if value is not None]
    return csv_dict


def sanitize_csv_datasets(csv_dict: dict, raw_dir: Path, out_dir: Path, out_subdir: str = None,
                          echo: Callable = print) -> list:
    datasets_input = []
    for i, (raw_data_dir, dataset_dir, dataset_name, first_image, processed_data_dir, output_dir, output_subdir) in (
            enumerate(zip(csv_dict['dataset']['raw_data_dir'], csv_dict['dataset']['dataset_dir'],
                          csv_dict['dataset']['dataset_name'], csv_dict['dataset']['first_image'],
                          csv_dict['dataset']['processed_data_dir'], csv_dict['dataset']['output_dir'],
                          csv_dict['dataset']['output_subdir']))):

        # Initialize via DiffractionDataset.from_image class method
        if first_image:
            first_image = Path(first_image)
            if not raw_data_dir:
                raw_data_dir = raw_dir
            # If the first image is not an absolute path, append the raw_data_dir and dataset_dir
            if not first_image.is_absolute():
                if not raw_data_dir:
                    echo(f'Image on line {i + 1} is a relative path and a \'raw_data_dir\' was not provided or a '
                         f'global raw data directory was not specified with --raw-dir', style='red')
                    raise typer.Abort()
                raw_data_dir = Path(raw_data_dir).resolve()
                if dataset_dir:
                    first_image = raw_data_dir / dataset_dir / first_image
                else:
                    first_image = raw_data_dir / first_image

        # Initialize via DiffractionDataset.__init__ method
        else:
            if not dataset_name:
                echo(f'Dataset on line {i + 1} does not have attribute \'dataset_name\'', style='red')
                raise typer.Abort()
            if not dataset_dir:
                echo(f'Dataset on line {i + 1} does not have attribute \'dataset_dir\'', style='red')
                raise typer.Abort()

            # If no raw_data directory is specified, use the global raw directory
            if not raw_data_dir:
                raw_data_dir = raw_dir
            if not raw_data_dir:
                echo(f'Dataset on line {i + 1} does not have the attribute \'raw_data_dir\' and a global '
                     f'raw data directory was not specified with --raw-dir', style='red')
                raise typer.Abort()
            raw_data_dir = Path(raw_data_dir).resolve()

        # If no processed_data directory is specified, use the global output directory
        if not processed_data_dir:
            if not out_dir:
                echo(f'Dataset on line {i + 1} does not have the attribute \'processed_data_dir\' and a global '
                     f'output directory was not specified with --out-dir', style='red')
                raise typer.Abort()
            else:
                processed_data_dir = out_dir
        processed_data_dir = Path(processed_data_dir).resolve()

        # If no output subdirectory is specified, use the global output subdirectory
        if not output_subdir:
            output_subdir = out_subdir

        if output_subdir:
            # Remove leading slashes and dots
            if output_subdir[:2] in ['./', '.\\']:
                output_subdir = output_subdir[2:]
            elif output_subdir[0] in ['/', '\\']:
                output_subdir = output_subdir[1:]

        datasets_input.append([raw_data_dir, dataset_dir, dataset_name, first_image, processed_data_dir, output_dir,
                               output_subdir])
    return datasets_input


def merge_configs(csv_dict: dict, dataset_index: int, **params):
    config = csv_dict['config']
    if dataset_index >= len(config['unit_cell']):
        return {}
    config = {key: value[dataset_index] for key, value in config.items()}

    config = copy.deepcopy(config)
    for key, value in params.items():
        # Skip unknown parameters
        if key not in config.keys():
            continue
        # Do not override values from the csv file
        existing_value = config[key]
        if value is not None and existing_value is None:
            config[key] = value

    # Reduce the config to only changes from default values
    config = {key: value for key, value in config.items() if value is not None}
    return config


def get_directory_size(directory: Path) -> int:
    return sum(f.stat().st_size for f in directory.rglob('*'))