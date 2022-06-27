import matplotlib.pyplot as plt
import numpy as np

from xtl.spectroscopy.base import Spectrum
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
        self.data.y_label = 'Absorption'

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

    def plot(self, baseline=False):
        '''
        Plot spectrum

        :param baseline: plot baseline
        :return:
        '''
        plt.figure()
        plt.plot(self.data.x, self.data.y, label=self.dataset)

        if baseline and hasattr(self.data, 'baseline'):
            plt.plot(self.data.baseline.x, self.data.baseline.y, ':k')

        ax = plt.gca()
        ax.set_xlabel(self.data.x_label)
        ax.set_ylabel(self.data.y_label)
