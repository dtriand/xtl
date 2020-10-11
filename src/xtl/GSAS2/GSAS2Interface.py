from xtl import cfg

import os
import sys
import click

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
                print(f'Invalid folder: {os.path.abspath(gsas_path)}')
            else:
                print('Failed to locate GSAS-II installation. Exiting...')
                raise FileNotFoundError

import GSASIIpwd as G2pd
import GSASIIlattice as G2lat
import GSASIIfiles as G2fil
import GSASIIspc as G2spc
import GSASIImath as G2m

working_directory = os.getcwd()
xtl_directories = {
    'maps': 'maps',
    'models': 'models',
    'reflections': 'reflections'
}


def _path_wrap(path):
    return os.path.join(working_directory, path)

