from pathlib import Path

import pytest


@pytest.fixture
def temp_files(request, tmp_path_factory):
    # Get marker
    marker = request.node.get_closest_marker("make_temp_files")
    params = getattr(request, 'param', None)

    if marker:
        files = marker.args
    else:
        if params is None:
            raise ValueError('No \'images\' marker found and no \'param\' provided on request')
        elif isinstance(params, list | tuple):
            files = request.param
        else:
            files = (request.param,)

    # Create a temporary directory to store the files
    temp_dir = tmp_path_factory.mktemp('cached')

    # Create all files and directories
    temp_files: list[Path] = []
    for f in files:
        if f is None:
            temp_files.append(None)
            continue
        f = temp_dir / f
        if (not f.suffix) and (f.name != ''):  # check if f is a directory because f.is_dir() fails when f doesn't exist
            f.mkdir(parents=True, exist_ok=True)
            print(f'Created empty temporary directory: {f}')
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
            print(f'Created empty temporary file: {f}')
        temp_files.append(f)

    # Return the temporary files as Path objects
    if len(temp_files) == 1:
        return temp_files[0]
    return temp_files
