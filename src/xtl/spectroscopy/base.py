from copy import deepcopy
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
        self.dataset = ''
        self.data: SpectrumData

        filename = kwargs.get('filename', None)
        if filename:
            self.from_file(**kwargs)

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
        obj.data = SpectrumData()
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
        self.data = SpectrumData()
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

    def export(self, filename: str or Path, file_fmt: str = 'csv'):
        '''
        Save spectrum to file.

        :param filename: output file
        :param file_fmt: file format
        :return:
        '''
        f = Path(filename)

        if file_fmt not in self.SUPPORTED_EXPORT_FMTS:
            raise InvalidArgument(raiser='file_fmt',
                                  message=f'{file_fmt}. Must be one of: {", ".join(self.SUPPORTED_EXPORT_FMTS)}')

        if file_fmt == 'csv':
            self._export_csv(filename)

    def _export_csv(self, filename: Path):
        '''
        CSV exporter.

        :param filename: output file
        :return:
        '''
        header = f'Dataset: {self.dataset}\n' \
                 f'{self.data.x_label}, {self.data.y_label}'
        data = np.vstack((self.data.x, self.data.y)).T
        np.savetxt(fname=filename, X=data, delimiter=',', header=header)

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

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y += other
            return self
        else:
            raise TypeError

    def __iadd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y += other
            return self
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y -= other
            return self
        else:
            raise TypeError

    def __isub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y -= other
            return self
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y *= other
            return self
        else:
            raise TypeError

    def __imul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y *= other
            return self
        else:
            raise TypeError

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y /= other
            return self
        else:
            raise TypeError

    def __itruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.data.y /= other
            return self
        else:
            raise TypeError

    def plot(self):
        '''
        Plot spectrum.

        :return:
        '''
        plt.figure()
        plt.plot(self.data.x, self.data.y, label=self.dataset)

    def add_to_plot(self):
        '''
        Add spectrum to existing plot. Must call ``matplotlib.pyplot.show()`` afterwards.

        :return:
        '''
        ax = plt.gca()
        ax.plot(self.data.x, self.data.y, label=self.dataset)

class SpectrumPoint:
    ...

class SpectrumCollection:

    SUPPORTED_IMPORT_FMTS = ['csv', 'cary_50_csv']

    def __init__(self):
        '''
        Representation of a library of Spectrums.
        '''
        self.spectra = {}

    def add_spectrum(self, spectrum: str or Path or Spectrum, label: str, **import_kwargs):
        '''
        Add a spectrum to library.

        :param spectrum: spectrum to add
        :param label: spectrum name
        :param import_kwargs: additional arguments to be passed at Spectrum.from_file()
        :return:
        '''
        if isinstance(spectrum, Spectrum):
            self.spectra[label] = spectrum
        elif isinstance(spectrum, str) or isinstance(spectrum, Path):
            self.spectra[label] = Spectrum(spectrum, **import_kwargs)
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
        if csv_fmt == 'xyy':
            for i in range(data.shape[0]):
                if i == 0:
                    continue
                self.spectra[f'Dataset {i}'] = Spectrum.from_data(x=data[0], y=data[1])
                self.spectra[f'Dataset {i}'].dataset = f'Dataset {i}'
        elif csv_fmt == 'xyxy':
            i = 0
            j = 1
            while i <= data.shape[0] - 1:
                self.spectra[f'Dataset {j}'] = Spectrum.from_data(x=data[i], y=data[i+1])
                self.spectra[f'Dataset {j}'].dataset = f'Dataset {j}'
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
        try:
            self._import_csv(f, import_kwargs)
        finally:
            f.unlink()

    def __iter__(self):
        # Iterate over self.spectra entries
        if not hasattr(self, '__iter'):
            self.__iter = self.spectra.__iter__()
        return self

    def __next__(self):
        if not self.spectra:
            raise StopIteration
        return self.spectra[next(self.__iter)]

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

    def set_labels(self, labels: list or tuple):
        '''
        Change the name of the stored datasets

        :param labels: new labels
        :return:
        '''
        if len(labels) != len(self.spectra):
            raise InvalidArgument(raiser='labels', message=f'Must be an iterable of length {len(self.spectra)}, not '
                                                           f'{len(labels)}')
        for old, new in zip(self.labels, labels):
            self.spectra[new] = self.spectra.pop(old)
            self.spectra[new].dataset = new
