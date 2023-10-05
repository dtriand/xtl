from pathlib import Path

import h5py
from tabulate import tabulate
import typer

from xtl.cli.cliio import CliIO


app = typer.Typer(add_completion=False)


def make_slice(expr: str):
    if expr:
        parts = expr.split(':')
        if len(parts) == 1:
            # slice(stop)
            parts = [None, parts[0]]
        # else: slice(start, stop[, step])
    else:
        # slice()
        parts = []
    return slice(*[int(p) if p else None for p in parts])


def get_formatted_dtype(obj):
    dtype = str(obj.dtype)
    if dtype.startswith('|S'):  # convert string dtypes
        string_length = dtype.split('|S')[-1]
        dtype = f'string{string_length}'
    return dtype


@app.command('h5peek', help='Explore the data structure of HDF5 files')
def cli_h5peek(fname: Path = typer.Argument(metavar='FILE', help='HDF5 file to inspect'),
               fpath: str = typer.Argument(metavar='PATH', default='/', help='Path directive'),
               dslice: str = typer.Argument(metavar='SLICE', default=None, help='[EXPERIMENTAL!] Slice operator for '
                                                                                'retrieving data')):
    # Check file and file format
    cli = CliIO()
    if not fname.exists():
        cli.echo('File does not exist', level='error')
        raise typer.Abort()
    if fname.suffix != '.h5' and fname.suffix != '.hdf5':
        cli.echo('Only .h5 and .hdf5 files can be read', level='error')
        raise typer.Abort()

    # Read file
    f = h5py.File(fname, 'r')

    # Test if an attribute is included in the given path
    if '.' in fpath:
        path, attr = fpath.split('.')
        if path.split('/')[-1] == '':
            cli.echo(f'Path cannot be None. Is there a trailing \'/\' on path? path=\'{path}\' '
                     f'attribute=\'{attr}\'', level='error')
            raise typer.Abort()
    else:
        path, attr = fpath, None
    if path not in f:
        cli.echo(f'Object \'{path}\' does not exist.', level='error')
        f.close()
        raise typer.Abort()

    # Report object structure
    cli.echo(f'{fname.name}{fpath}')
    try:
        if attr:  # Reading of attributes
            a = f[path].attrs.get(attr)

            if a is None:  # check if attribute exists
                cli.echo(f'No attribute \'{attr}\' under path \'{path}\'', level='error')
                raise typer.Abort()

            dtype = get_formatted_dtype(a)
            cli.echo(f'ATTRIBUTE')
            cli.echo(f'  dtype: {dtype}')
            cli.echo(f'  shape: {a.shape}')
            cli.echo(f'  bytes: {a.nbytes}')
            cli.echo(f'  value: {a[:]}')

        elif isinstance(f[path], h5py._hl.dataset.Dataset):  # Reading of datasets
            obj = f[path]
            cli.echo(f'DATASET')
            cli.echo(f'  dtype: {obj.dtype}')
            cli.echo(f'  shape: {obj.shape}')
            cli.echo(f'  bytes: {obj.nbytes}')

            if dslice is None:  # default slice for scalars
                slice_str = ()
            else:  # slice from user
                slice_str = make_slice(dslice)
            cli.echo(f'  value: {obj[slice_str]}')

            attrs = f[path].attrs
            cli.echo('ATTRIBUTES')
            cli.echo('\n'.join([f'  .{key}' for key in attrs.keys()]))

        else:  # Reading of groups
            objs = f[path].values()
            last_item = len(objs) - 1
            tree = []
            for i, obj in enumerate(objs):
                entry = ''
                if i < last_item:
                    entry += '├ '
                else:
                    entry += '└ '
                entry += '/' + obj.name.split('/')[-1]  # print child's name only

                entry_type = 'unk'
                if isinstance(obj, h5py._hl.group.Group):
                    entry_type = 'group'
                elif isinstance(obj, h5py._hl.dataset.Dataset):
                    shape = '×'.join([f'{dim}' for dim in obj.shape])
                    ndim = obj.ndim
                    dtype = get_formatted_dtype(obj)
                    if ndim >= 1:  # value is an array
                        entry_type = f'{dtype}: {shape}'
                    else:  # value is a scalar
                        entry_type = f'{dtype}'
                else:
                    entry_type = type(obj)
                    print(entry_type)
                tree.append([entry, f'[{entry_type}]'])

            attrs = f[path].attrs
            for i, attr_name in enumerate(attrs.keys()):
                a = attrs.get(attr_name)
                dtype = get_formatted_dtype(a)
                tree.append([f'.{attr_name}', f'[{dtype}]'])
            cli.echo(tabulate(tree, tablefmt='plain'))
    except KeyError:
        parent, child = path.rsplit('/', maxsplit=1)
        cli.echo(f'Object \'{child}\' does not exist under path \'{parent}\'', level='error')
        raise typer.Abort()
    finally:
        f.close()


if __name__ == '__main__':
    app()
