from functools import partial
from math import floor
from pathlib import Path

import fabio
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pyFAI


class Image:


    def __init__(self):
        self.file: Path = None
        self.fmt: str = None
        self.frame: int = None
        self.header_only: bool = False
        self._fabio: fabio.fabioimage.FabioImage = None
        self._pyfai: pyFAI.AzimuthalIntegrator = None

        self.cmap = 'inferno'
        self.cmap_bad_values = 'white'
        self.mask_color = (0, 1, 0)
        self.mask_alpha = 0.5


    def open(self, file: str or Path, frame: int = None):
        self.file = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(self.file)
        self.frame = frame
        self._fabio = fabio.open(self.file, self.frame)
        self.fmt = self._fabio.classname


    def openheader(self, file: str or Path):
        self.file = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(self.file)
        self._fabio = fabio.openheader(self.file)
        self.fmt = self._fabio.classname
        self.header_only = True


    def save(self):
        ...


    @property
    def data(self):
        return self._fabio.data


    def load_geometry(self, file: str or Path):
        file = Path(file)
        self._pyfai = pyFAI.load(str(file))


    def load_mask(self, array):
        ...


    def azimuthal_integration_cake(self, **kwargs):
        if not self._pyfai:
            raise Exception('No geometry information available. Run load_geometry() method first.')
        _data = kwargs.get('_data', self.data)
        center = self._pyfai.poni2 / self._pyfai.pixel2, self._pyfai.poni1 / self._pyfai.pixel1  # in pixels
        x_size, y_size = self.data.shape
        pixels_from_center = x_size - center[0], y_size - center[0]
        npt_rad = kwargs.get('npt_rad', floor(max(pixels_from_center)))
        npt_azim = kwargs.get('npt_azim', 360)
        error_model = kwargs.get('error_model', 'poisson')
        mask = kwargs.get('mask', None)  # self.mask
        dummy = kwargs.get('dummy', None)
        unit = kwargs.get('unit', 'q_A^-1')
        if unit in ['2theta', 'tth', '2th']:
            unit = '2th_deg'
        elif unit in ['q']:
            unit = 'q_A^-1'
        filename = kwargs.get('filename', None)

        print(pixels_from_center, npt_rad)
        cake = self._pyfai.integrate2d(data=_data, npt_rad=npt_rad, npt_azim=npt_azim, filename=filename,
                                       error_model=error_model, mask=mask, dummy=dummy, unit=unit)
        # cake = intensities, 2theta (radial angle), chi (azimuthal angle), optional intensities sigma if error_model
        #   was supplied (3 or 4-length tuple)
        return cake


    def azimuthal_integration_1d(self, **kwargs):
        if not self._pyfai:
            raise Exception('No geometry information available. Run load_geometry() method first.')
        center = self._pyfai.poni2 / self._pyfai.pixel2, self._pyfai.poni1 / self._pyfai.pixel1  # in pixels
        x_size, y_size = self.data.shape
        pixels_from_center = x_size - center[0], y_size - center[0]
        npt = kwargs.get('npt', floor(max(pixels_from_center)))
        error_model = kwargs.get('error_model', 'poisson')
        mask = kwargs.get('mask', None)  # self.mask
        dummy = kwargs.get('dummy', None)
        unit = kwargs.get('unit', 'q_A^-1')
        if unit in ['2theta', 'tth', '2th']:
            unit = '2th_deg'
        elif unit in ['q']:
            unit = 'q_A^-1'
        filename = kwargs.get('filename', None)

        print(pixels_from_center, npt)

        hist = self._pyfai.integrate1d(data=self.data, npt=npt, filename=filename, error_model=error_model, mask=mask,
                                       dummy=dummy, unit=unit)
        # hist = 2theta (radial angle), intensities, optional intensities sigma if error_model was supplied
        #   (2 or 3-length tuple)
        return hist


    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fig = kwargs.pop('fig', plt.gcf())
        norm = kwargs.pop('norm', partial(matplotlib.colors.Normalize, clip=False))
        if norm in ['log', 'log10']:
            norm = partial(matplotlib.colors.LogNorm, clip=False)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)

        cmap = matplotlib.cm.get_cmap(self.cmap)
        cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)

        # pos = ax.imshow(self.data, cmap=cmap, norm=norm(vmin=0, vmax=1e9))
        pos = ax.imshow(self.data, cmap=cmap, norm=norm(vmin=vmin, vmax=vmax))
        if self._pyfai:
            # why is this in reverse?
            center_in_pixels = self._pyfai.poni2 / self._pyfai.pixel2, self._pyfai.poni1 / self._pyfai.pixel1
            ax.scatter(*center_in_pixels, marker='X', s=70, facecolors='red', edgecolors='white')

        # fig.colorbar(pos, label='Intensity')
        ax.set_title(self.file.name)


    def plot_cake(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fig = kwargs.pop('fig', plt.gcf())
        norm = kwargs.pop('norm', partial(matplotlib.colors.Normalize, clip=False))
        if norm in ['log', 'log10']:
            norm = partial(matplotlib.colors.LogNorm, clip=False)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        mask = kwargs.pop('mask', None)
        overlay_mask = kwargs.pop('overlay_mask', None)
        xlabel = kwargs.pop('xlabel', 'Radial angle / 2\u03b8 (\u00b0)')

        cmap = matplotlib.cm.get_cmap(self.cmap)
        cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)

        integrator = partial(self.azimuthal_integration_cake, error_model=None, **kwargs)
        intensities, ttheta, chi = integrator(_data=self.data, mask=mask)
        extent = (ttheta.min(), ttheta.max(), chi.min(), chi.max())
        pos = ax.imshow(intensities, origin='lower', extent=extent, cmap=cmap, aspect='auto', interpolation='nearest',
                        norm=norm(vmin=vmin, vmax=vmax))

        if overlay_mask:
            intensities_masked = integrator(_data=~mask, mask=None)[0]
            intensities_masked = np.clip(intensities_masked, 0, 1/1e9)
            mask_color = matplotlib.colors.to_rgba(self.mask_color)
            cmap_mask = matplotlib.colors.LinearSegmentedColormap.from_list('mask', [mask_color, (1, 1, 1, 0)], N=2)
            ax.imshow(intensities_masked, origin='lower', extent=(ttheta.min(), ttheta.max(), chi.min(), chi.max()),
                      cmap=cmap_mask, aspect='auto', interpolation='nearest', alpha=self.mask_alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Azimuthal angle / \u03c7 (\u00b0)')



    def plot_1d(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fig = kwargs.pop('fig', plt.gcf())
        norm = kwargs.pop('norm', partial(matplotlib.colors.Normalize, clip=False))
        if norm in ['log', 'log10']:
            norm = partial(matplotlib.colors.LogNorm, clip=False)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        xlabel = kwargs.pop('xlabel', 'Radial angle / 2\u03b8 (\u00b0)')
        ylabel = kwargs.pop('ylabel', 'Intensity (arbitrary units)')

        cmap = matplotlib.cm.get_cmap(self.cmap)
        cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)

        ttheta, intensities = self.azimuthal_integration_1d(error_model=None, **kwargs)
        pos = ax.plot(ttheta, intensities)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)