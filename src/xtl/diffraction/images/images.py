import copy
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path

import fabio
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm, to_rgba, LinearSegmentedColormap
from matplotlib.image import AxesImage
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
from pyFAI.geometry import Geometry
from pyFAI.containers import Integrate1dResult, Integrate2dResult

import xtl  # for type hinting
from xtl.diffraction.images.masks import detector_masks


class Image:

    def __init__(self):
        # I/O options
        self.file: Path = None
        self.fmt: str = None
        self.frame: int = 0
        self.header_only: bool = False

        # Masking options
        self.mask: ImageMask = None
        self.masked_pixels_value = np.nan

        # Readers and integrators
        self._fabio: fabio.fabioimage.FabioImage = None
        self.geometry: Geometry = None
        self.ai1: 'xtl.diffraction.images.integrators.AzimuthalIntegrator1D' = None
        self.ai2: 'xtl.diffraction.images.integrators.AzimuthalIntegrator2D' = None

        # Intensity data
        self._data: np.array = None  # Is not None when the raw data has been modified
        self.is_summed: bool = False
        self.summed_frames: list = []
        self.no_summed_frames: int = 0

        # Reading of i.e. CBF images, where the individual frames are separate files
        self._is_multifile: bool = None  # True when frames are saved as separate files, False for i.e. H5 files
        self._filename_template: str = ''  # The common substring of a file
        self._file_ext: str = ''  # File extension, useful when working with compressed images (eg .cbf.gz)
        self._frames_digits: int = None  # Number of digits after self._filename_template
        self._no_frames: int = None  # Number of frames as determined by a glob search
        self._current_frame: int = 0  # The frame number as determined by the filename

        # Plotting options
        self.cmap = 'inferno'
        self.cmap_bad_values = 'white'
        self.mask_color = (0, 1, 0)
        self.mask_alpha = 0.5
        self.detector_image_origin = 'upper'

    def open(self, file: str or Path, frame: int = 0, is_eager: bool = True):
        self.file = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(self.file)
        self.frame = frame
        self._fabio = fabio.open(self.file, self.frame)
        self.fmt = self._fabio.classname
        if self.fmt == 'EigerImage':
            self._is_multifile = False
        else:
            self._is_multifile = True

        if self._is_multifile and is_eager:
            self._determine_multifile_frames()
            print(f'Found {self.no_frames} frames.')
            if self.frame != self._current_frame and self.frame <= self.no_frames:
                # self._current_frame is read from the filename, while self.frame is provided upon the function call
                print(f'Eagerly loading frame {self.frame} instead of frame {self._current_frame} which was provided '
                      f'as an input file.')
                self._current_frame = self.frame
                self.file = self.file.parent / self._get_frame_name(self._current_frame)
                self._fabio = fabio.open(self.file, self.frame)
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
        if self._data is None:  # if raw data has not been modified
            return self._fabio.data
        return self._data  # for modified raw data

    @property
    def data_masked(self):
        m = np.ma.masked_where(~self.mask.data, self.data)
        if m.dtype != 'float64':
            m = m.astype('float')
        m.fill_value = self.masked_pixels_value
        return m

    def _determine_multifile_frames(self):
        """
        Determines the number of frames for multi-filed image formats (e.g. CBF).
        """
        self._file_ext = self.file.suffix
        if self._file_ext == '.gz':  # deal with compressed images
            self._file_ext = Path(self.file.name.split('.gz')[0]).suffix + '.gz'
        # Glob all files with the same starting character in the filename and same extension
        tree = list(self.file.parent.glob(f'{self.file.name[0]}*{self._file_ext}'))
        tree.sort()

        # Find the longest match between the first and last filenames
        fname0 = tree[0].name.split(self._file_ext)[0]
        fname1 = tree[-1].name.split(self._file_ext)[0]
        match = SequenceMatcher(None, fname0, fname1).find_longest_match()

        # Save list of frames
        if match.a == match.b == 0:
            self._filename_template = fname0[match.a:match.a+match.size]
            self._frames_digits = len(fname0[match.a+match.size:])
            # Assuming frames are 1-indexed
            self._no_frames = int(fname1[match.b+match.size:])
            self._current_frame = int(self.file.name.split(self._file_ext)[0].replace(self._filename_template, '')) - 1

    def _get_frame_name(self, frame_no):
        """
        Returns the filename for a given frame number. If the image is a multi-file format (e.g. CBF images), the new
        filename is constructed. Otherwise (e.g. H5 images), the original filename is returned.
        """
        if self._is_multifile:
            return self._filename_template + str(frame_no + 1).zfill(self._frames_digits) + self._file_ext
        return self.file.name

    @property
    def no_frames(self):
        """
        The number of frames available in the stack.
        """
        if self._is_multifile:
            return self._no_frames
        return self._fabio.nframes

    def next_frame(self):
        """
        Loads the next frame in the image stack.
        """
        self.frame += 1
        if self.frame >= self.no_frames:
            raise Exception('Run out of frames!')
        if self._is_multifile:
            self.file = self.file.parent / self._get_frame_name(self.frame)
            self._fabio = fabio.open(self.file, self.frame)
        else:
            self._fabio = self._fabio.next()

    def previous_frame(self):
        """
        Loads the previous frame in the image stack.
        """
        self.frame -= 1
        if self.frame == 0:
            raise Exception('Already at the first frame!')
        if self._is_multifile:
            self.file = self.file.parent / self._get_frame_name(self.frame)
            self._fabio = fabio.open(self.file, self.frame)
        else:
            self._fabio = self._fabio.previous()

    def get_frame(self, frame_no: int):
        """
        Loads the requested frame from the stack.
        """
        self.frame = frame_no
        if self.frame >= self.no_frames:
            raise Exception(f'Image contains only {self.no_frames} frames.')
        if self._is_multifile:
            self.file = self.file.parent / self._get_frame_name(self.frame)
            self._fabio = fabio.open(self.file, self.frame)
        else:
            self._fabio.get_frame(frame_no)

    def sum_frames(self, no_frames: int):
        """
        Sum the intensities of a given number of frames. If no_frames='all', then the summation is performed on all
        remaining frames in the stack.
        """
        if no_frames == 'all':
            no_frames = self.no_frames - self.frame
        elif no_frames > (self.no_frames - self.frame):
            raise Exception(f'Image contains only {self.no_frames} frames. Current frame: {self.frame}')
        data = self.data
        self.summed_frames.append(self.frame)
        for _ in range(no_frames - 1):
            self.next_frame()
            self.summed_frames.append(self.frame)
            data += self.data
        self._data = data
        self.is_summed = True
        self.no_summed_frames = len(self.summed_frames)

    def load_geometry(self, data: str or Path or dict):
        """
        Load pyFAI geometry from a .poni file or a PONI-like dictionary.
        """
        self.geometry = Geometry()
        if isinstance(data, str) or isinstance(data, Path):
            file = Path(data)
            self.geometry.load(str(file))
        elif isinstance(data, dict):
            self.geometry.set_config(data)

    def save_geometry(self, filename):
        """
        Save geometry to .poni file.
        """
        self.check_geometry()
        f = Path(filename)
        f.unlink(missing_ok=True)
        self.geometry.save(f)

    @property
    def has_geometry(self):
        if not self.geometry:
            return False
        return True

    def check_geometry(self):
        if not self.geometry:
            raise Exception('No geometry information available. Run load_geometry() method first.')
        return True

    @property
    def beam_center(self):
        """
        The location of the beam center (x, y) in pixel coordinates.
        """
        self.check_geometry()
        return self.geometry.poni2 / self.geometry.pixel2, self.geometry.poni1 / self.geometry.pixel1

    @property
    def dimensions(self):
        """
        The total number of pixels along x and y in the image.
        """
        return self.data.shape[::-1]

    def make_mask(self):
        self.mask = ImageMask(nx=self._fabio.shape[-2], ny=self._fabio.shape[-1], parent=self)
        return self.mask

    def initialize_azimuthal_integrator(self, dim: int = 1):
        """
        Initialize an azimuthal integrator with 1 or 2 dimensions.

        :param dim: No. of dimensions for integration
        :return: AzimuthalIntegrator1D or AzimuthalIntegrator2D
        :raises ValueError: When ``dim`` is not 1 or 2
        """
        if dim == 1:
            from xtl.diffraction.images.integrators import AzimuthalIntegrator1D
            self.ai1 = AzimuthalIntegrator1D(image=self)
            return self.ai1
        elif dim == 2:
            from xtl.diffraction.images.integrators import AzimuthalIntegrator2D
            self.ai2 = AzimuthalIntegrator2D(image=self)
            return self.ai2
        else:
            raise ValueError('Invalid number of dimensions. Must be 1 or 2.')

    def azimuthal_integration_1d(self, **kwargs) -> Integrate1dResult:
        """
        Perform a 1D azimuthal integration.

        :param dict kwargs: Options for ``xtl.diffraction.images.AzimuthalIntegrator1D.initialize()``
        :return:
        """
        if self.ai1 is None:
            self.initialize_azimuthal_integrator(dim=1)
        self.ai1.initialize(**kwargs)
        self.ai1.integrate()
        return self.ai1.results

    def azimuthal_integration_2d(self, **kwargs) -> Integrate2dResult:
        """
        Perform a 2D azimuthal integration (aka cake projection).

        :param dict kwargs: Options for ``xtl.diffraction.images.AzimuthalIntegrator2D.initialize()``
        :return:
        """
        if self.ai2 is None:
            self.initialize_azimuthal_integrator(dim=2)
        self.ai2.initialize(**kwargs)
        self.ai2.integrate()
        return self.ai2.results

    def plot(self, ax: plt.Axes = None, fig: plt.Figure = None, xlabel: str = None, ylabel: str = None,
             title: str = None, zscale: str = None, zmin: float = None, zmax: float = None, cmap: str = None,
             bad_value_color: str = None, apply_mask: bool = True, overlay_mask: bool = False) \
            -> tuple[plt.Axes, plt.Figure, AxesImage]:
        """
        Prepare a plot of the recorded intensity data. ``plt.show()`` must be called separately to display the plot.

        :param matplotlib.axes.Axes ax: Axes instance to draw into
        :param matplotlib.figure.Figure fig: Figure instance to draw into
        :param str xlabel: x-axis label (default: None)
        :param str ylabel: y-axis label (default: None)
        :param str title: Plot title (default: filename)
        :param str zscale: z-axis scale, one from: ``'linear'``, ``'log'`` (default: ``'linear'``)
        :param float zmin: z-axis minimum value
        :param float zmax: z-axis maximum value
        :param str cmap: A Matplotlib colormap name to be used as the intensity scale
        :param str bad_value_color: The missing values color
        :param bool apply_mask: Whether to apply the image mask prior to plotting
        :param bool overlay_mask: Whether to overlay the image mask on top of the integration results
        :return:
        """
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        if title is None:
            title = self.file.name
        if zscale in [None, 'linear']:
            norm = partial(Normalize, clip=False)
        elif zscale in ['log', 'log10']:
            norm = partial(LogNorm, clip=False)
        else:
            raise ValueError(f'Invalid value for \'zscale\'. Must be one of: linear, log')

        if cmap is None:
            cmap = get_cmap(self.cmap)
        else:
            cmap = get_cmap(cmap)
        if bad_value_color is None:
            cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)
        else:
            cmap.set_bad(color=bad_value_color, alpha=1.0)

        if apply_mask:
            data = self.data_masked
        else:
            data = self.data

        img = ax.imshow(data, cmap=cmap, norm=norm(vmin=zmin, vmax=zmax), origin=self.detector_image_origin)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if self.geometry:
            ax.scatter(*self.beam_center, marker='X', s=70, facecolors='red', edgecolors='white')
        if overlay_mask:
            mask_cmap = LinearSegmentedColormap.from_list(name='mask', N=2,
                                                          colors=[to_rgba(self.mask_color), (1, 1, 1, 0)])
            ax.imshow(self.mask.data, cmap=mask_cmap, alpha=self.mask_alpha, origin=self.detector_image_origin,
                      vmin=0, vmax=1)
        return ax, fig, img

    def plot_1d(self, **kwargs):
        """
        Prepare a plot of the results of a 1D azimuthal integration.

        :param dict kwargs: Options for ``xtl.diffraction.images.AzimuthalIntegrator1D.plot()``
        :return:
        """
        if self.ai1 is None:
            raise Exception('Azimuthal integrator not initialized yet. '
                            'Run initialize_azimuthal_integrator(dim=1) first.')
        return self.ai1.plot(**kwargs)

    def plot_cake(self, **kwargs):
        """
        Prepare a plot of the results of a 2D azimuthal integration.

        :param dict kwargs: Options for ``xtl.diffraction.images.AzimuthalIntegrator2D.plot()``
        :return:
        """
        if self.ai2 is None:
            raise Exception('Azimuthal integrator not initialized yet. '
                            'Run initialize_azimuthal_integrator(dim=2) first.')
        return self.ai2.plot(**kwargs)


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

    def mask_detector(self, detector: str or dict, gaps=True, frame=True, double_pixels=True):
        if isinstance(detector, str) and detector not in detector_masks.keys():
            raise Exception(f'No mask available for detector {detector}. Choose one from: '
                            f'{", ".join(detector_masks.keys())}')
        if isinstance(detector, dict):
            mask = detector
        else:
            mask = detector_masks[detector]

        gaps_mask = mask.get('gaps', None)
        frame_mask = mask.get('frame', None)
        double_pixels_mask = mask.get('double_pixels', None)
        for apply_mask, mask in zip((gaps, frame, double_pixels), (gaps_mask, frame_mask, double_pixels_mask)):
            if not apply_mask or not mask:
                continue
            for row in mask.get('rows', {}):
                self.mask_rows(*row)
            for col in mask.get('cols', {}):
                self.mask_cols(*col)

    def mask_blemishes(self, fname: Path = None, text: str = '', blist: list = None, zero_indexed=True):
        blemishes = []
        if fname:
            text += Path(fname).read_text()
        if text:
            for line in text.split('\n'):
                if line.startswith('#'):
                    continue
                x, y = line.split(',')
                x, y = int(x), int(y)
                blemishes.append([x, y])
        if blist:
            blemishes += [[x, y] for x, y in blist]

        for x, y in blemishes:
            if not zero_indexed:  # For non-zero indexed pixel coordinates (i.e. from Matlab)
                x -= 1
                y -= 1
            self.mask_pixel(x, y)
