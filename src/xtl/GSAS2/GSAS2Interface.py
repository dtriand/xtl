import sys
from pathlib import Path
from typing import Union

import click

from xtl import cfg

try:
    gsas_path = cfg['dependencies']['gsas'].value
    sys.path.append(gsas_path)
    import GSASIIscriptable as G2sc
except ModuleNotFoundError:
    print('This operation requires a functional installation of GSAS-II.\n'
          'GSAS-II can be downloaded from: https://subversion.xray.aps.anl.gov/trac/pyGSAS')

    path_invalid = True
    attempt = 1

    while path_invalid and attempt <= 3:
        gsas_path = click.prompt("Please specify the GSAS-II installation folder "
                                 "(should contain 'GSASIIscriptable.py')")
        try:
            sys.path.append(gsas_path)
            import GSASIIscriptable as G2sc
            # Save the path to the config if the import was successful
            cfg.set('dependencies', 'gsas', gsas_path)
            cfg.save()
            path_invalid = False  # break loop
        except ModuleNotFoundError:
            attempt += 1
            if attempt <= 3:
                print(f'Invalid folder: {Path(gsas_path).absolute()}')
            else:
                print('Failed to locate GSAS-II installation. Exiting...')
                raise FileNotFoundError

import GSASIIpwd as G2pd
import GSASIIlattice as G2lat
import GSASIIfiles as G2fil
import GSASIIspc as G2spc
import GSASIImath as G2m


class Settings:
    def __init__(self):
        self._working_directory = Path.cwd()
        self._xtl_directories = {
            'maps': 'maps',
            'models': 'models',
            'reflections': 'reflections'
        }

    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, new_directory):
        wd = Path(new_directory)
        if wd.is_dir():
            self._working_directory = Path(new_directory)
        else:
            raise NotADirectoryError

    @property
    def xtl_directories(self):
        return self._xtl_directories


settings = Settings()


def _path_wrap(path: Union[str, Path]) -> Path:
    return settings.working_directory / path

