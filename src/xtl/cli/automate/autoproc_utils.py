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


def sanitize_csv_datasets(csv_dict: dict, raw_dir: Path, out_dir: Path, echo: Callable = print) -> list:
    datasets_input = []
    for i, (raw_data_dir, dataset_dir, dataset_name, first_image, processed_data_dir, output_dir) in (
            enumerate(zip(csv_dict['dataset']['raw_data_dir'], csv_dict['dataset']['dataset_dir'],
                          csv_dict['dataset']['dataset_name'], csv_dict['dataset']['first_image'],
                          csv_dict['dataset']['processed_data_dir'], csv_dict['dataset']['output_dir']))):
        if first_image:
            first_image = Path(first_image)
        if not dataset_name:
            echo(f'Dataset on line {i + 1} does not have attribute \'dataset_name\'', style='red')
            raise typer.Abort()
        if not dataset_dir:
            echo(f'Dataset on line {i + 1} does not have attribute \'dataset_dir\'', style='red')
            raise typer.Abort()
        if not raw_data_dir:
            if not raw_dir:
                echo(f'Dataset on line {i + 1} does not have the attribute \'raw_data_dir\' and a global '
                          f'raw data directory was not specified with --raw-dir', style='red')
                raise typer.Abort()
            else:
                raw_data_dir = raw_dir
            raw_data_dir = Path(raw_data_dir)
        if not processed_data_dir:
            if not out_dir:
                echo(f'Dataset on line {i + 1} does not have the attribute \'processed_data_dir\' and a global '
                          f'output directory was not specified with --out-dir', style='red')
                raise typer.Abort()
            else:
                processed_data_dir = out_dir
            processed_data_dir = Path(processed_data_dir)
        if not first_image.is_absolute():
            if raw_data_dir:
                if dataset_dir:
                    first_image = raw_data_dir / dataset_dir / first_image
                else:
                    first_image = raw_data_dir / first_image
        datasets_input.append([raw_data_dir, dataset_dir, dataset_name, first_image, processed_data_dir, output_dir])
    return datasets_input