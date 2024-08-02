import os
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from xtl.exceptions import InvalidArgument


class SpectrumData:

    def __init__(self):
        '''
        Store spectral data.
        '''
        self.x: np.ndarray
        self.y: np.ndarray
        self.x_label = ''
        self.y_label = ''

    def check_data(self):
        '''
        Perform various checks on the stored data.

        :return:
        '''
        # Check for descending order in wavelengths
        if self.x[0] > self.x[1]:
            self.x = np.flip(self.x)
            self.y = np.flip(self.y)

class Spectrum:

    SUPPORTED_IMPORT_FMTS = ['csv', 'cary_50_csv']
    SUPPORTED_EXPORT_FMTS = ['csv']

    def __init__(self, **kwargs):
        '''
        Representation of a spectrum.

        :param kwargs:
        '''
        self.file: Path
        self._dataset = ''
        self._original_dataset = ''
        self.data = SpectrumData()

        filename = kwargs.get('filename', None)
        if filename:
            self.from_file(**kwargs)
        self._post_init()

    def _post_init(self):
        '''
        Runs after initialization

        :return:
        '''
        pass

    @property
    def dataset(self):
        '''
        Dataset label

        :return:
        '''
        return self._dataset

    @dataset.setter
    def dataset(self, new_label):
        if not self._dataset:
            self._original_dataset = new_label
        self._dataset = new_label

    def from_file(self, filename: str or Path, file_fmt: str, dataset_name: str = '', **import_kwargs):
        '''
        Load a spectrum from a file. Supports only ``.csv`` files.

        :param filename: file to load
        :param file_fmt: file type
        :param dataset_name: name of the dataset
        :param import_kwargs: additional arguments for numpy.loadtxt()
        :return:
        '''
        self.file = Path(filename)
        if not self.file.exists():
            raise FileNotFoundError(self.file)

        if file_fmt not in self.SUPPORTED_IMPORT_FMTS:
            raise InvalidArgument(raiser='file_fmt',
                                  message=f'{file_fmt}. Must be one of: {", ".join(self.SUPPORTED_IMPORT_FMTS)}')

        if dataset_name:
            self.dataset = dataset_name
        else:
            self.dataset = self.file.name

        if file_fmt == 'csv':
            self._import_csv(self.file, import_kwargs)
        elif file_fmt == 'cary_50_csv':
            self._import_cary_50_csv(self.file, import_kwargs)
        self._post_init()

    @classmethod
    def from_data(cls, x: list or tuple or np.ndarray, y: list or tuple or np.ndarray):
        '''
        Load a spectrum from data.

        :param x: x-values (wavelengths)
        :param y: y-values (intensities)
        :return:
        '''
        x = np.array(x)
        y = np.array(y)
        for i in (x, y):
            if len(i.shape) != 1:
                raise InvalidArgument(raiser=i, message=f'Must be a 1D array')
        obj = cls()
        obj.data.x, obj.data.y = x, y
        obj.data.check_data()
        return obj

    def _import_csv(self, filename: Path, import_kwargs=dict()):
        '''
        CSV importer.

        :param filename: file to load
        :param import_kwargs: additional arguments for numpy.loadtxt()
        :return:
        '''
        delimiter = import_kwargs.get('delimiter', ',')
        skiprows = import_kwargs.get('skiprows', 0)
        data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)

        # Check data shape
        if data.shape[0] == 2:  # (2, N)
            pass
        elif data.shape[1] == 2:  # (N, 2)
            data = data.T
        else:
            raise InvalidArgument(raiser='filename',
                                  message=f'Dataset must have dimensions (2, N) or (N, 2), not: {data.shape}')

        # Initialize SpectrumData object
        self.data.x = data[0]
        self.data.y = data[1]
        self.data.check_data()

    def _import_cary_50_csv(self, filename: Path, import_kwargs=dict()):
        '''
        Importer for ``.csv`` files from Cary 50.

        :param filename: file to load
        :param import_kwargs: additional arguments for numpy.loadtxt()
        :return:
        '''
        f = filename.parent / (filename.stem + '_temp.csv')
        f.write_text(filename.read_text().split('\n\n')[0] + '\n')
        import_kwargs['skiprows'] = 2
        try:
            self._import_csv(f, import_kwargs)
        finally:
            f.unlink()

    def export(self, filename: str or Path, file_fmt: str = 'csv', **export_kwargs):
        '''
        Save spectrum to file.

        :param filename: output file
        :param file_fmt: file format
        :param export_kwargs: additional arguments for numpy.savetxt()
        :return:
        '''
        f = Path(filename)

        if file_fmt not in self.SUPPORTED_EXPORT_FMTS:
            raise InvalidArgument(raiser='file_fmt',
                                  message=f'{file_fmt}. Must be one of: {", ".join(self.SUPPORTED_EXPORT_FMTS)}')

        if file_fmt == 'csv':
            self._export_csv(filename, **export_kwargs)

    def _export_csv(self, filename: Path, **export_kwargs):
        '''
        CSV exporter.

        :param filename: output file
        :param export_kwargs: additional arguments for numpy.savetxt()
        :return:
        '''
        if filename.suffix == '':
            filename = filename.parent / (filename.stem + '.csv')

        header = f'Dataset: {self.dataset}\n' \
                 f'{self.data.x_label}, {self.data.y_label}'
        data = np.vstack((self.data.x, self.data.y)).T

        fmt = export_kwargs.pop('fmt', ('%f', '%f'))
        np.savetxt(fname=filename, X=data, delimiter=',', header=header, fmt=fmt, **export_kwargs)

    def _find_nearest_index(self, value: int or float, vector: np.ndarray):
        return np.abs(vector - value).argmin()

    def __getitem__(self, item):
        if isinstance(item, slice):
            # Slice SpectrumData based on wavelength (eg. SpectrumData()[200:800])
            i1, i2 = self._find_nearest_index(item.start, self.data.x), self._find_nearest_index(item.stop, self.data.x)
            if i1 > i2:  # reverse indices if passed in descending order
                i1, i2 = i2, i1
            sp = deepcopy(self)
            sp.data.x = sp.data.x[i1:i2]
            sp.data.y = sp.data.y[i1:i2]
            return sp
        elif isinstance(item, tuple):
            raise NotImplementedError
            # if len(item) != 2:
            #     raise TypeError('Invalid argument type')
            # print('multidim: ', item)
        else:
            raise NotImplementedError  # SpectrumData()[280] -> SpectrumPoint()
            # print('plain: ', item)

    def __iter__(self):
        # Iterate over datapoints
        if not hasattr(self, '__iter'):
            self.__iter = self.data.x.__iter__(), self.data.y.__iter__()
        return self

    def __next__(self):
        if not hasattr(self.data, 'x') or not hasattr(self.data, 'y'):
            raise StopIteration
        xi, yi = self.__iter
        return next(xi), next(yi)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y += other
            return self
        else:
            raise TypeError

    def __iadd__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y += other
            return self
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y -= other
            return self
        else:
            raise TypeError

    def __isub__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y -= other
            return self
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y *= other
            return self
        else:
            raise TypeError

    def __imul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y *= other
            return self
        else:
            raise TypeError

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y /= other
            return self
        else:
            raise TypeError

    def __itruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            self.data.y /= other
            return self
        else:
            raise TypeError

    def plot(self, **kwargs):
        '''
        Plot spectrum.

        :param kwargs: additional arguments for matplotlib.pyplot.plot()
        :return:
        '''
        label = kwargs.pop('label', self.dataset)
        plt.figure()
        plt.plot(self.data.x, self.data.y, label=label, **kwargs)

    def add_to_plot(self, **kwargs):
        '''
        Add spectrum to existing plot. Must call ``matplotlib.pyplot.show()`` afterwards.

        Additional keyword arguments:
        - ax: matplotlib.axes.Axes / axes instance to plot data on

        :param kwargs: additional arguments for matplotlib.pyplot.plot()
        :return:
        '''
        label = kwargs.pop('label', self.dataset)
        ax = kwargs.pop('ax', plt.gca())
        ax.plot(self.data.x, self.data.y, label=label, **kwargs)
        ax.set_xlabel(self.data.x_label)
        ax.set_ylabel(self.data.y_label)

class SpectrumPoint:
    ...

class SpectrumCollection:

    SUPPORTED_IMPORT_FMTS = ['csv', 'cary_50_csv']

    def __init__(self, spectrum_type=Spectrum):
        '''
        Representation of a library of Spectrums.

        :param spectrum_type: load spectrums as this class, must be a subclass of xtl.spectroscopy.base.Spectrum
        '''
        self.spectra = {}

        if not issubclass(spectrum_type, Spectrum):
            raise InvalidArgument(raiser='spectrum_type', message='Must be Spectrum or a subclass of Spectrum')
        self._spectrum_class = spectrum_type

    def add_spectrum(self, spectrum: str or Path or Spectrum, label: str, **import_kwargs):
        '''
        Add a spectrum to library.

        :param spectrum: spectrum to add
        :param label: spectrum name
        :param import_kwargs: additional arguments to be passed at Spectrum.from_file()
        :return:
        '''
        if isinstance(spectrum, self._spectrum_class):
            self.spectra[label] = spectrum
        elif isinstance(spectrum, str) or isinstance(spectrum, Path):
            self.spectra[label] = self._spectrum_class(spectrum, **import_kwargs)
        else:
            raise InvalidArgument(raiser='spectrum',
                                  message='Must be of type str, pathlib.Path or xtl.spectroscopy.Spectrum')

    def import_file(self, filename: str or Path, file_fmt: str, **import_kwargs):
        '''
        Import multiple spectra from a single file.

        :param filename: file to load
        :param file_fmt: file format
        :param import_kwargs: additional arguments to be passed at numpy.loadtxt()
        :return:
        '''
        if file_fmt not in self.SUPPORTED_IMPORT_FMTS:
            raise InvalidArgument(raiser='file_fmt',
                                  message=f'{file_fmt}. Must be one of: {", ".join(self.SUPPORTED_IMPORT_FMTS)}')

        filename = Path(filename)
        if file_fmt == 'csv':
            self._import_csv(filename, import_kwargs)
        elif file_fmt == 'cary_50_csv':
            self._import_cary_50_csv(filename, import_kwargs)

    def _import_csv(self, filename: Path, import_kwargs=dict()):
        '''
        CSV importer.

        Additional kwargs:
        - x_axis: 'vertical' or 'horizontal' / The axis on which the values propagate
        - csv_fmt: 'xyy' or 'xyxy' / How data is stored in the file (x -> wavelength, y -> intensities)

        :param filename: file to load
        :param import_kwargs: additional arguments for numpy.loadtxt()
        :return:
        '''
        delimiter = import_kwargs.get('delimiter', ',')
        skiprows = import_kwargs.get('skiprows', 0)

        data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)

        x_axis = import_kwargs.get('x_axis', 'vertical')
        if x_axis == 'vertical':
            data = data.T
        elif x_axis == 'horizontal':
            pass
        else:
            raise InvalidArgument(raiser='x_axis', message='Must be vertical or horizontal')

        csv_fmt = import_kwargs.get('csv_fmt', 'xyy')
        cols = data.shape[0]
        if csv_fmt == 'xyy':
            for i in range(cols):
                if i == 0:
                    continue
                dname = f'{filename.name}:d{i}' if cols > 2 else filename.name
                self.spectra[dname] = self._spectrum_class.from_data(x=data[0], y=data[i])
                self.spectra[dname].dataset = dname
                self.spectra[dname].file = filename
        elif csv_fmt == 'xyxy':
            i = 0
            j = 1
            while i <= cols - 1:
                dname = f'{filename.name}:d{j}' if cols > 2 else filename.name
                self.spectra[dname] = self._spectrum_class.from_data(x=data[i], y=data[i+1])
                self.spectra[dname].dataset = dname
                self.spectra[dname].file = filename
                i += 2
                j += 1
        else:
            raise InvalidArgument(raiser='csv_fmt', message='Must be xyy or xyxy')

    def _import_cary_50_csv(self, filename: Path, import_kwargs=dict()):
        '''
        Importer for ``.csv`` files from Cary 50.

        :param filename: file to load
        :param import_kwargs: additional arguments for numpy.loadtxt()
        :return:
        '''
        f = filename.parent / (filename.stem + '_temp.csv')
        data = filename.read_text()
        data = data.replace(',\n', '\n')  # remove trailing commas
        data = data.split('\n\n')[0] + '\n'  # remove instrument comments at the end of the file
        f.write_text(data)
        import_kwargs['skiprows'] = 2
        import_kwargs['csv_fmt'] = 'xyxy'
        keep_original_dataset_names = import_kwargs.get('keep_original_dataset_names', True)
        try:
            self._import_csv(f, import_kwargs)
            if keep_original_dataset_names:
                dnames = data.split('\n')[0].split(',,')
                self.set_labels([dname.rstrip(',') for dname in dnames], _change_originals=True)
        finally:
            f.unlink()

    def import_directory(self, dirname: str or Path, file_fmt='csv', prefix='', suffix='', recursive=False,
                         **import_kwargs):
        '''
        Load all files from a single directory. Looks for pattern ``dirname/{prefix}*{suffix}.{ext}`` using
        glob.glob()

        :param dirname: directory to load files from
        :param file_fmt: file format to look for
        :param prefix: prefix for search string
        :param suffix: suffix for search string
        :param recursive: look in subdirectories
        :param import_kwargs: additional arguments to be passed at numpy.loadtxt()
        :return:
        '''
        dirname = Path(dirname)
        if not dirname.exists():
            raise FileNotFoundError(dirname)
        if not dirname.is_dir():
            raise InvalidArgument(raiser='dirname', message='Must be a directory')
        if file_fmt not in self.SUPPORTED_IMPORT_FMTS:
            raise InvalidArgument(raiser='file_fmt',
                                  message=f'{file_fmt}. Must be one of: {", ".join(self.SUPPORTED_IMPORT_FMTS)}')

        ext = ''
        if file_fmt in ['csv', 'cary_50_csv']:
            ext = 'csv'
        else:
            raise InvalidArgument(raiser=file_fmt, message='Unsupported file format')

        # Find files
        old_dir = Path.cwd()
        os.chdir(dirname)  # glob.glob() searches only in current directory in Python < 3.10
        search_string = f'{prefix}*{suffix}.{ext}'
        if recursive:
            search_string = '**/' + search_string
        files = glob(pathname=search_string)
        os.chdir(old_dir)

        # Import files
        for f in files:
            fname = dirname / f
            with open(fname, 'r') as fp:  # skip reading of summary csv's generated by xtl
                if fp.readline() == '# xtl_summary_csv':
                    continue
            try:
                self.import_file(filename=fname, file_fmt=file_fmt, **import_kwargs)
            except Exception as e:
                print(f'Error while loading file: {fname}')
                raise e

    def export(self, dirname: str or Path, subdirs=False, prefix='', suffix='', sequential_naming=False, summary=False,
               **export_kwargs):
        '''
        Export all datasets to files. By default datasets are labelled based on their original filename and dataset
        label. This can be changed by argument ``sequential_naming``.

        :param dirname: directory to export files to
        :param subdirs: export files to separate subdirectories
        :param prefix: prefix for filenames
        :param suffix: suffix for filenames
        :param sequential_naming: labelling datasets using numbers instead of dataset names
        :param summary: export summary csv
        :param export_kwargs: additional arguments for numpy.savetxt()
        :return:
        '''
        dirname = Path(dirname)
        summary_data = ['# xtl_summary_csv', 'File,Dataset,Label,NewFileName']

        imax = len(self)
        dmax = len(str(imax))  # number of digits
        for i, spectrum in enumerate(self):
            if sequential_naming:
                fname = f'{prefix}_{str(i+1).zfill(dmax)}' if prefix else str(i+1).zfill(dmax)
            else:
                fname = f'{prefix}_{spectrum.dataset.replace(" ", "_")}' if prefix \
                    else spectrum.dataset.replace(" ", "_")

            child_dir = dirname if not subdirs else dirname / fname
            child_dir.mkdir(parents=True, exist_ok=True)
            f = child_dir / (fname + f'_{suffix}' if suffix else fname)
            spectrum.export(filename=f, file_fmt='csv', **export_kwargs)

            summary_data.append(f'{spectrum.file},{spectrum._original_dataset},{spectrum.dataset},{f}')

        if summary:
            summary_csv = dirname / f'{prefix + "_" if prefix else ""}summary.csv'
            summary_csv.write_text('\n'.join(summary_data))


    def __len__(self):
        return len(self.spectra)

    def __iter__(self):
        # Iterate over self.spectra entries
        if not hasattr(self, '__iter'):
            self.__iter = self.spectra.__iter__()
        return self

    def __next__(self):
        if not self.spectra:
            raise StopIteration
        return self.spectra[next(self.__iter)]

    def __getitem__(self, item):
        if isinstance(item, int):
            return tuple(self.spectra.values())[item]
        elif isinstance(item, str):
            return self.spectra[item]
        else:
            raise NotImplementedError

    @property
    def labels(self):
        '''
        Datasets names
        :return:
        '''
        return list(self.spectra.keys())

    @property
    def datasets(self):
        '''
        Datasets names
        :return:
        '''
        return self.labels

    def set_labels(self, labels: list or tuple, **kwargs):
        '''
        Change the name of the stored datasets

        :param labels: new labels
        :return:
        '''
        if len(labels) != len(self.spectra):
            raise InvalidArgument(raiser='labels', message=f'Must be an iterable of length {len(self.spectra)}, not '
                                                           f'{len(labels)}')

        reindex = kwargs.get('_reindex', [])  # the order in which to read the provided labels
        change_originals = kwargs.get('_change_originals', False)
        if not reindex:
            # Standard relabelling
            for old, new in zip(self.labels, labels):
                self.spectra[new] = self.spectra.pop(old)
                self.spectra[new].dataset = new
                if change_originals:
                    self.spectra[new]._original_dataset = new
        else:
            # Relabelling using a reindex list
            old_labels = self.labels
            for i, j in zip(np.argsort(reindex), range(len(old_labels))):
                old = old_labels[i]  # choose dataset to rename based on reindex array
                new = labels[j]  # apply new labels sequentially
                self.spectra[new] = self.spectra.pop(old)
                self.spectra[new].dataset = new
                if change_originals:
                    self.spectra[new]._original_dataset = new
