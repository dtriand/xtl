import copy
from functools import partial
from math import floor
from pathlib import Path

import fabio
import matplotlib.cm
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import pyFAI

from xtl.diffraction.images.masks import detector_masks


class Image:

    def __init__(self):
        self.file: Path = None
        self.fmt: str = None
        self.frame: int = 0
        self.header_only: bool = False
        self.mask: ImageMask = None
        self._fabio: fabio.fabioimage.FabioImage = None
        self._pyfai: pyFAI.AzimuthalIntegrator = None

        self.cmap = 'inferno'
        self.cmap_bad_values = 'white'
        self.mask_color = (0, 1, 0)
        self.mask_alpha = 0.5
        self.detector_image_origin = 'upper'

    def open(self, file: str or Path, frame: int = 0):
        self.file = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(self.file)
        self.frame = frame
        self._fabio = fabio.open(self.file, self.frame)
        self.fmt = self._fabio.classname
        self.mask = self.make_mask()

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

    @property
    def data_masked(self):
        m = np.ma.masked_where(~self.mask.data, self._fabio.data)
        if m.dtype != 'float64':
            m = m.astype('float')
        m.fill_value = np.nan
        return m

    @property
    def no_frames(self):
        return self._fabio.nframes

    def next_frame(self):
        self._fabio = self._fabio.next()
        self.frame += 1

    def previous_frame(self):
        self._fabio = self._fabio.previous()
        self.frame -= 1

    def load_geometry(self, file: str or Path):
        file = Path(file)
        self._pyfai = pyFAI.load(str(file))

    @property
    def beam_center(self):
        return self._pyfai.poni2 / self._pyfai.pixel2, self._pyfai.poni1 / self._pyfai.pixel1

    def make_mask(self):
        self.mask = ImageMask(nx=self._fabio.shape[-2], ny=self._fabio.shape[-1], parent=self)
        return self.mask

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
        mask = kwargs.get('mask', self.mask)
        if isinstance(mask, ImageMask):
            mask = mask.data
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
        mask = kwargs.get('mask', self.mask)
        if mask is None:
            mask = self.make_mask().data
        elif isinstance(mask, ImageMask):
            mask = mask.data
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
        mask = kwargs.get('mask', self.mask)
        if isinstance(mask, ImageMask):
            mask = mask.data
        overlay_mask = kwargs.pop('overlay_mask', None)

        cmap = matplotlib.cm.get_cmap(self.cmap)
        cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)

        m = ax.imshow(self.data, cmap=cmap, norm=norm(vmin=vmin, vmax=vmax), origin=self.detector_image_origin)
        if self._pyfai:
            # why is this in reverse?
            center_in_pixels = self._pyfai.poni2 / self._pyfai.pixel2, self._pyfai.poni1 / self._pyfai.pixel1
            ax.scatter(*center_in_pixels, marker='X', s=70, facecolors='red', edgecolors='white')
        if overlay_mask:
            mask_color = matplotlib.colors.to_rgba(self.mask_color)
            mask_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mask', [mask_color, (1, 1, 1, 0)], N=2)
            ax.imshow(self.mask.data, cmap=mask_cmap, alpha=self.mask_alpha, origin=self.detector_image_origin,
                      vmin=0, vmax=1)

        # fig.colorbar(pos, label='Intensity')
        title = kwargs.pop('title', self.file.name)
        ax.set_title(title)
        return m

    def plot_cake(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fig = kwargs.pop('fig', plt.gcf())
        norm = kwargs.pop('norm', partial(matplotlib.colors.Normalize, clip=False))
        if norm in ['log', 'log10']:
            norm = partial(matplotlib.colors.LogNorm, clip=False)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        mask = kwargs.get('mask', self.mask)
        if mask is None:
            mask = self.make_mask().data
        elif isinstance(mask, ImageMask):
            mask = mask.data
        overlay_mask = kwargs.pop('overlay_mask', None)
        xlabel = kwargs.pop('xlabel', 'Radial angle / 2\u03b8 (\u00b0)')

        cmap = matplotlib.cm.get_cmap(self.cmap)
        cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)

        integrator = partial(self.azimuthal_integration_cake, error_model=None, **kwargs)
        intensities, ttheta, chi = integrator(_data=self.data, mask=~mask)
        m = ax.imshow(intensities, origin='lower', extent=(ttheta.min(), ttheta.max(), chi.min(), chi.max()),
                      cmap=cmap, aspect='auto', interpolation='nearest', norm=norm(vmin=vmin, vmax=vmax))

        if overlay_mask:
            intensities_masked, ttheta, chi = integrator(_data=mask, mask=None)   # mask: True = keep, False = discard
            intensities_masked = np.clip(intensities_masked, 0, 1/1e24)  # 0: discard, 1/1e24: keep
            mask_color = matplotlib.colors.to_rgba(self.mask_color)
            mask_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mask', [mask_color, (1, 1, 1, 0)], N=2)
            ax.imshow(intensities_masked, origin='lower', extent=(ttheta.min(), ttheta.max(), chi.min(), chi.max()),
                      cmap=mask_cmap, aspect='auto', interpolation='nearest', alpha=self.mask_alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Azimuthal angle / \u03c7 (\u00b0)')
        return m

    def plot_1d(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fig = kwargs.pop('fig', plt.gcf())
        norm = kwargs.pop('norm', partial(matplotlib.colors.Normalize, clip=False))
        if norm in ['log', 'log10']:
            norm = partial(matplotlib.colors.LogNorm, clip=False)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        mask = kwargs.get('mask', self.mask)
        if mask is None:
            mask = self.make_mask().data
        elif isinstance(mask, ImageMask):
            mask = mask.data
        xlabel = kwargs.pop('xlabel', 'Radial angle / 2\u03b8 (\u00b0)')
        ylabel = kwargs.pop('ylabel', 'Intensity (arbitrary units)')

        cmap = matplotlib.cm.get_cmap(self.cmap)
        cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)

        ttheta, intensities = self.azimuthal_integration_1d(error_model=None, mask=~mask, **kwargs)
        pos = ax.plot(ttheta, intensities)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


class ImageMask:

    def __init__(self, nx: int, ny: int, parent: Image = None):
        self.nx = nx
        self.ny = ny
        self._data: np.ndarray
        self._initialize_empty_mask()
        self.parent: Image = parent

    def _initialize_empty_mask(self):
        self._data = np.ones((self.nx, self.ny), dtype=bool)  # array of True's

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    def mask_pixel(self, x: int, y: int):
        self._data[y, x] = False

    def mask_rows(self, i: int, j: int):
        self._data[i:j+1, :] = False

    def mask_cols(self, i: int, j: int):
        self._data[:, i:j+1] = False

    def mask_rectangle(self, pt1: tuple, pt2: tuple):
        x1, y1 = pt1
        x2, y2 = pt2
        self._data[y1:y2+1, x1:x2+1] = False

    def mask_polygon(self, pts: tuple):
        vertices = [(y, x) for x, y in pts]
        polygon = matplotlib.path.Path(vertices)

        x, y = np.mgrid[0:self.nx, 0:self.ny]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        masked_points = ~polygon.contains_points(coords)

        mask = masked_points.reshape(self.nx, self.ny)
        self._data &= mask

    def mask_circle(self, center: tuple, radius: float):
        circle = matplotlib.path.Path.circle(center, radius)

        x, y = np.mgrid[0:self.nx, 0:self.ny]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        masked_points = ~circle.contains_points(coords)

        mask = masked_points.reshape(self.nx, self.ny)
        self._data &= mask

    def mask_intensity_greater_than(self, value: float):
        if not self.parent:
            raise Exception
        self._data[self.parent.data > value] = False

    def mask_intensity_less_than(self, value: float):
        if not self.parent:
            raise Exception
        self._data[self.parent.data < value] = False

    def __invert__(self):
        new_mask = copy.deepcopy(self)
        new_mask._data = ~new_mask._data
        return new_mask

    def invert(self):
        self._data = ~self._data

    def mask_detector(self, detector: str, gaps=True, frame=True, double_pixels=True):
        if detector not in detector_masks.keys():
            raise Exception(f'No mask available for detector {detector}. Choose one from: '
                            f'{", ".join(detector_masks.keys())}')

        gaps_mask = detector_masks[detector].get('gaps', None)
        frame_mask = detector_masks[detector].get('frame', None)
        double_pixels_mask = detector_masks[detector].get('double_pixels', None)
        for apply_mask, mask in zip((gaps, frame, double_pixels), (gaps_mask, frame_mask, double_pixels_mask)):
            if not apply_mask or not mask:
                continue
            for row in mask.get('rows', {}):
                self.mask_rows(*row)
            for col in mask.get('cols', {}):
                self.mask_cols(*col)

