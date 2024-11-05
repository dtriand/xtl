from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def make_temp_files(request, tmp_path_factory):
    # Create a tuple of temp files to create
    if hasattr(request.param, '__iter__'):
        files = request.param
    else:
        files = (request.param,)

    # Create a temporary directory to store the files
    temp_dir = tmp_path_factory.mktemp('cached')
    temp_files: list[Path] = []

    # Create all files and directories
    for f in files:
        if f is None:
            temp_files.append(None)
            continue
        f = temp_dir / f
        if f.is_dir():
            f.mkdir(parents=True, exist_ok=True)
            print(f'Created empty temporary directory: {f}')
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
            print(f'Created empty temporary file: {f}')
        temp_files.append(f)

    # Return the temporary files as Path objects
    return temp_files

