from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from xtl.spectroscopy.base import Spectrum
from xtl.math.filtering import moving_average
from xtl.exceptions import InvalidArgument


class Baseline:

    BASELINE_TYPES = ['zero', 'flat']

    def __int__(self):
        self.x: np.ndarray
        self.y: np.ndarray
        self.type = 'flat'

    def __bool__(self):
        if not hasattr(self, 'y'):
            return False
        return True


class AbsorptionSpectrum(Spectrum):

    def __init__(self, **kwargs):
        '''
        Representation of a UV-vis absorption spectrum.

        :param kwargs:
        '''
        super().__init__(**kwargs)
        self.data.baseline = Baseline()

    def _post_init(self):
        self.data.x_label = 'Wavelength (nm)'
        self.data.y_label = 'Absorbance'

    def find_baseline(self, search_region: list or tuple):
        '''
        Determine a baseline by averaging the absorbance values in ``search_region``.

        :param search_region: wavelength range for deremination of baseline
        :return:
        '''
        if not isinstance(search_region, list) and not isinstance(search_region, tuple):
            raise InvalidArgument(raiser='search_region', message='Must be an iterable')
        if len(search_region) != 2:
            raise InvalidArgument(raiser='search_region', message='Must be an iterable of length 2')

        wv1, wv2 = search_region
        for i, _ in enumerate((wv1, wv2)):
            if not isinstance(_, int) and not isinstance(_, float):
                raise InvalidArgument(raiser='search_region', message=f'Element {i} is not a number')

        # Check if wavelengths are passed in reverse
        if wv1 > wv2:
            wv1, wv2 = wv2, wv1

        # Baseline calculation
        baseline = np.average(self[wv1:wv2].data.y)

        # Store results
        self.data.baseline.x = self.data.x
        self.data.baseline.y = np.ones_like(self.data.x) * baseline

    def subtract_baseline(self):
        '''
        Subtract a pre-calculated baseline.

        :return:
        '''
        if self.data.baseline:
            self.data.y - self.data.baseline.y
            self.data.baseline.y = np.zeros_like(self.data.baseline.x)
            self.data.baseline.type = 'zero'
        else:
            raise Exception('Baseline not initialized. Run find_baseline() first.')

    def normalize(self, eliminate_negatives=False):
        '''
        Normalize absorbance values.

        :param eliminate_negatives: remove negative values first
        :return:
        '''
        if eliminate_negatives:
            self.data.y += np.abs(self.data.y.min())
        self.data.y /= self.data.y.max()
        self.data.y_label = 'Normalized absorbance'

    def smooth_data(self, func: callable = moving_average, inplace=False, **func_kwargs):
        '''
        Apply smoothing function to data.

        :param func: function to apply. Must have signature func(x: np.ndarray, **kwargs)
        :param inplace: replace original data with smoothed data
        :param func_kwargs: additional arguments to pass to smoothing function
        :return: new AbsorptionSpectrum instance with smoothed data if inplace=True
        '''
        if not callable(func):
            raise InvalidArgument(raiser='func', message='Must be a function or callable')

        if inplace:
            obj = self
        else:
            obj = deepcopy(self)

        obj.data.y = func(obj.data.y, **func_kwargs)
        if not inplace:
            return obj

    def plot(self, baseline=False, **kwargs):
        '''
        Plot spectrum

        :param baseline: plot baseline
        :param kwargs: additional arguments for matplotlib.pyplot.plot()
        :return:
        '''
        plt.figure()
        plt.plot(self.data.x, self.data.y, label=self.dataset, **kwargs)

        if baseline and hasattr(self.data, 'baseline'):
            plt.plot(self.data.baseline.x, self.data.baseline.y, ':k')

        ax = plt.gca()
        ax.set_xlabel(self.data.x_label)
        ax.set_ylabel(self.data.y_label)
