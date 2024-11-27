from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig


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