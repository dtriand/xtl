from pathlib import Path

import gemmi
import numpy as np

from xtl import cfg
from xtl.exceptions import InvalidArgument, FileError
from xtl import math as xm
from .components import PhaseMixture
from .parameters import InstrumentalParameters
from ..GSAS2 import GSAS2Interface as GI
from ..GSAS2 import objects as GO


class Project(GI.G2sc.G2Project):

    def __init__(self, filename, debug=False):
        self.debug = debug
        if GI._path_wrap(filename).exists():
            super().__init__(gpxfile=GI._path_wrap(filename))
        else:
            super().__init__(newgpx=GI._path_wrap(filename))
        self.filename = Path(self.filename)
        self._directory, self._name = self.filename.parent, self.filename.name

    def _backup_gpx(self):
        backup_dir = GI.settings.working_directory / '.xtl'
        backup_gpx = backup_dir / self._name
        if not backup_dir.exists():
            backup_dir.mkdir()
        import shutil
        shutil.copy2(src=self.filename, dst=backup_gpx)
        if self.debug:
            print(f'Backing up .gpx file at {backup_gpx}')

    def _get_gpx_version(self):
        """
        Finds the last project.bakXX.gpx file in the folder and returns XX + 1. If no .bak.gpx file is found in the
        directory, returns 0. Can find files up to .bak9999.gpx

        :return:
        """
        filename = self.filename.stem
        bak = []
        for i in range(1, 4):
            bak += [f for f in self._directory.glob(f'{filename}.bak{"?" * i}.gpx')]  # bak?, bak??, bak???, bak????

        if not bak:  # No bak files found
            return 0

        last_file = sorted(bak)[-1]

        # Get the number after .bak
        file_version = last_file.stem.split('.bak')[1]
        file_version = int(file_version) + 1
        return file_version

    @staticmethod
    def _prepare_directory(directory):
        path = GI.settings.working_directory / directory
        if not path.exists():
            path.mkdir()

    def _get_phases_and_histograms_iterator(self, phase, histogram):
        """
        Returns an iterable of phases and an iterable of histograms. If no phases/histograms are given, all
        phases/histograms of the project are returned.

        :param GI.G2sc.G2Phase phase:
        :param GI.G2sc.G2PwdrData histogram:
        :return: ([G2Phase], [G2PwdrData])
        :rtype: Tuple[List[GI.G2sc.G2Phase], List[GI.G2sc.G2PwdrData]]
        """

        if phase:
            self.check_is_phase(phase)
            phases = [phase]
        else:
            phases = self.phases()

        if histogram:
            self.check_is_histogram(histogram)
            histograms = [histogram]
        else:
            histograms = self.histograms()

        return phases, histograms

    def add_comment(self, comment):
        from datetime import datetime
        xtl_vers = cfg['xtl']['version'].value
        now = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
        self.data['Notebook']['data'].append(f"xtl {xtl_vers} @ {now}\n{comment}")

    def add_phase(self, file, name=None, histograms=[], type=''):
        if not name:
            name = Path(file).stem  # grab filename from a full path
        allowed_types = ['macromolecular', 'small_molecule']
        if type not in allowed_types:
            raise InvalidArgument(raiser='phase_type', message=f"Unknown phase type '{type}'\n"
                                                               f"Choose one from: {','.join(allowed_types)}")
        elif type == 'macromolecular':
            return self.add_phase_macromolecular(GI._path_wrap(file), name, histograms)
        elif type == 'small_molecule':
            print('Not implemented')
            exit(-1)

    def add_phase_macromolecular(self, file, name=None, histograms=[]):
        # Check for valid unit-cell
        space_group_string = ''
        with open(file, 'r') as fp:
            for i, line in enumerate(fp):
                if line.startswith('CRYST1'):
                    space_group_string = line[55:65]
                    break
            if not space_group_string:
                raise FileError(file=file, message='No CRYST1 record found.')

        # Space group validation
        valid, error = self.validate_space_group(space_group_string)
        if not valid:
            raise FileError(file=file, message=error)

        # PDB file validation (check for atoms)
        valid, error = self.validate_pdb_file(file)
        if not valid:
            raise FileError(file=file, message=error)

        # Add phase to project
        super().add_phase(phasefile=file, phasename=name, fmthint='PDB')
        return self.phase(phasename=name)

    @staticmethod
    def validate_space_group(space_group_string):
        """
        Checks whether a string is a valid space group.

        :param str space_group_string: String to be parsed
        :return: is_valid, error
        :rtype: tuple[bool, str]
        """
        SGError, SGData = GI.G2spc.SpcGroup(space_group_string)
        if SGError:  # If space group is valid, then SGError = 0
            return False, f"Invalid space group {SGData['SpGrp']}."
        return True, ''

    @staticmethod
    def validate_pdb_file(pdb_file):
        """
        Checks whether a .pdb file contains any ATOM records.

        :param str pdb_file: File path
        :return: is_valid, error
        :rtype: tuple[bool, str]
        """
        from imports.G2phase import PDB_ReaderClass
        reader = PDB_ReaderClass()
        if not reader.ContentsValidator(pdb_file):
            return False, f'No ATOM or HETATM records found.'
        return True, ''

    @staticmethod
    def check_is_phase(phase):
        if not isinstance(phase, GI.G2sc.G2Phase):
            raise InvalidArgument(message=f'{phase} is not a G2Phase object.')

    @staticmethod
    def check_is_histogram(histogram):
        if not isinstance(histogram, GI.G2sc.G2PwdrData):
            raise InvalidArgument(message=f'{histogram} is not a G2PwdrData object.')

    def check_is_phase_in_project(self, phase):
        self.check_is_phase(phase)
        if phase not in self.phases():
            raise InvalidArgument(message=f'Phase {phase} is not in project {self._name}.')

    def check_is_histogram_in_project(self, histogram):
        self.check_is_histogram(histogram)
        if histogram not in self.histograms():
            raise InvalidArgument(message=f'Histogram {histogram} is not in project {self._name}.')

    def get_spacegroup(self, phase):
        """
        Returns a Gemmi representation of a phase's space group.

        :param GI.G2sc.G2Phase phase:
        :return: Gemmi spacegroup object
        :rtype: gemmi.SpaceGroup
        """
        self.check_is_phase(phase)
        return gemmi.find_spacegroup_by_name(hm=phase.data['General']['SGData']['SpGrp'])

    def get_cell(self, phase):
        """
        Returns unit-cell parameters without the volume.

        :param GI.G2sc.G2Phase phase:
        :return: (a, b, c, alpha, beta, gamma)
        :rtype: tuple
        """
        self.check_is_phase(phase)
        return tuple(phase.get_cell().values())[:-1]

    def get_formula(self, phase):
        """
        Returns the phase's composition as a chemical formula (e.g. H12 C6 O6). The elements appear in order of
        ascending atomic number.

        :param phase:
        :return: Composition formula
        :rtype: str
        """
        self.check_is_phase(phase)
        formula = ''
        for element, count in phase.composition.items():
            formula += f'{element}{int(count)} '
        return formula

    def get_phase_type(self, phase):
        """
        Returns the phase type i.e. nuclear, magnetic, macromolecular or faulted.

        :param GI.G2sc.G2Phase phase:
        :return: nuclear, magnetic, macromolecular or faulted
        :rtype: str
        """
        self.check_is_phase(phase)
        return phase.data['General']['Type']

    def has_map(self, phase):
        """
        Checks if a map is stored inside the phase dict.

        :param GI.G2sc.G2Phase phase:
        :return: True if has map else False
        :rtype: bool
        """
        self.check_is_phase(phase)
        # If a map is present then rho = np.ndarray, else rho = ''
        return True if isinstance(phase['General']['Map']['rho'], np.ndarray) else False

    def get_wavelength(self, histogram):
        """
        Returns the wavelength of a histogram and whether it is from a lab source. If it is a lab source, it returns
        'Lam1' as the wavelength.

        :param GI.G2sc.G2PwdrData histogram:
        :return: (wavelength, is_lab)
        :rtype: tuple[float, bool]
        """
        self.check_is_histogram(histogram)
        iparams = histogram.InstrumentParameters
        if 'Lam' in iparams:
            return iparams['Lam'][1], False
        elif 'Lam1' in iparams:
            return iparams['Lam1'][1], True

    def get_data_range(self, histogram):
        """
        Returns the 2theta range of a raw histogram, not the refined range.

        :param GI.G2sc.G2PwdrData histogram:
        :return: (2theta_min, 2theta_max)
        :rtype: tuple[float, float]
        """
        self.check_is_histogram(histogram)
        ttheta = histogram.getdata('x')
        return ttheta[0], ttheta[-1]

    def get_histogram(self, histogram, subtract_background=False):
        """
        Returns histogram datapoints as numpy array. The following columns are included: [2theta, Io, sigma(Io), Ic,
        background]. If ``subtract_background=True``, the returned array has the following 4 columns instead [2theta,
        Io-background, sigma(Io), Ic-background].

        :param GI.G2sc.G2PwdrData histogram: Histogram to extract datapoints from
        :param bool subtract_background: Subtract the background from intensities
        :return: Datapoints array with columns: 2theta, Io, sigma(Io), Ic, background or 2theta, Io-background,
                 sigma(Io), Ic-background.
        :rtype: np.ndarray
        """
        self.check_is_histogram(histogram)
        ttheta = histogram.getdata('x').reshape(-1, 1)
        Io = histogram.getdata('yobs').reshape(-1, 1)
        sigmaIo = histogram.getdata('yweight').reshape(-1, 1)
        Ic = histogram.getdata('ycalc').reshape(-1, 1)
        background = histogram.getdata('background').reshape(-1, 1)
        # residual = histogram.getdata('residual').reshape(-1, 1)
        if subtract_background:
            datapoints = np.hstack((ttheta, Io - background, sigmaIo, Ic - background))
        else:
            datapoints = np.hstack((ttheta, Io, sigmaIo, Ic, background))
        return datapoints

    def get_reflections(self, phase, histogram, scale=False):
        """
        Returns a numpy array with the following columns: h, k, l, d-spacing, Fo^2, sigma(Fo^2), Fc^2, phase.
        Observed structure factor errors are computed as sqrt(Fo^2).

        :param GI.G2sc.G2Phase phase: Phase to extract reflections from
        :param GI.G2sc.G2PwdrData histogram: Histogram to extract reflections from
        :param bool scale: Scale structure factors larger than 1e7 (for .mtz files)
        :return: Reflections array with columns: h, k, l, d-spacing, Fo^2, sigma(Fo^2), Fc^2, phase
        :rtype: np.ndarray
        """
        self.check_is_phase(phase)
        self.check_is_histogram(histogram)

        gpx_reflections = histogram.reflections()[phase.name]['RefList']
        # Available columns in gpx_reflections
        # h, k, l, multiplicity, d-spacing, 2theta, sig, gam, Fosq, Fcsq, phase, Icorr, Prfo, Trans, ExtP
        # 0, 1, 2, 3,            4,         5,      6,   7,   8,    9,    10,    11,    12,   13,    14
        h = gpx_reflections[:, 0].reshape(-1, 1)  # .reshape(-1, 1) converts the 1D horizontal array to a 2D vertical
        k = gpx_reflections[:, 1].reshape(-1, 1)
        l = gpx_reflections[:, 2].reshape(-1, 1)
        d = gpx_reflections[:, 4].reshape(-1, 1)
        Fo2 = gpx_reflections[:, 8].reshape(-1, 1)
        Fo2_sigma = np.sqrt(Fo2)
        Fc2 = gpx_reflections[:, 9].reshape(-1, 1)
        phase = gpx_reflections[:, 10].reshape(-1, 1)
        reflections = np.hstack((h, k, l, d, Fo2, Fo2_sigma, Fc2, phase))

        if scale:
            # MTZ files can only save numbers < 1.e7
            # Fo^2 and Fc^2 can occasionally be larger, so they must be first scaled down before saving
            times_scaled = 0
            scale_factor = 10
            while True:
                has_larger_than_allowed_in_mtz = False
                column_max = np.amax(reflections[:, [4, 6]], axis=0)  # Fo^2, Fc^2
                for value in column_max:
                    if value > 1.e7:
                        has_larger_than_allowed_in_mtz = True
                if has_larger_than_allowed_in_mtz:
                    reflections[:, [4, 6]] /= scale_factor  # Fo^2, Fc^2
                    reflections[:, 5] /= np.sqrt(scale_factor)  # sigma(Fo^2)
                    times_scaled += 1
                    continue
                else:
                    break
            if times_scaled:
                hist_name = histogram.name.split('.')[0][5:]
                print(f"Scaling down structure factors for histogram {hist_name} by a factor of "
                      f"{round(scale_factor ** times_scaled, 3)}")
        return reflections

    def export_reflections(self, phase=None, histogram=None, filetype='cif', as_intensities=False, **kwargs):
        """
        Export structure factors for a specific phase and histogram to file. Supported filetypes are ``cif`` (mmCIF) and
        ``mtz``. If phase and/or histogram are ``None``, then reflections from all phases and histograms are exported.

        Keyword Arguments
            * ``pretty: bool = False`` -- Space-align tabular data in .cif files. Might result in a slower export.
            * ``include_histograms: bool = False`` -- Save histogram datapoints in .cif files.
            * ``mini_mtz: bool = False`` -- Export separate .mtz files for each histogram

        :param GI.G2sc.G2Phase or None phase: Phase to export reflections from
        :param GI.G2sc.G2PwdrData or None histogram: Histogram to export reflections from
        :param filetype: 'cif' or 'mtz'
        :param as_intensities: Export intensities instead of structure factors.
        :param kwargs: See below
        :return:
        """
        phases, histograms = self._get_phases_and_histograms_iterator(phase, histogram)

        if filetype == 'cif':
            pretty = kwargs.get('pretty', False)
            include_histograms = kwargs.get('include_histograms', False)
            self._export_reflections_as_cif(phases, histograms, pretty=pretty, as_intensities=as_intensities,
                                            include_histograms=include_histograms)
        elif filetype == 'mtz':
            mini_mtz = kwargs.get('mini_mtz', False)
            if mini_mtz:
                for histogram in histograms:
                    self._export_reflections_as_mtz(phases, [histogram], as_intensities=as_intensities)
            else:
                self._export_reflections_as_mtz(phases, histograms, as_intensities=as_intensities)
        else:
            supported_formats = ['cif', 'mtz']
            raise InvalidArgument(raiser='filetype', message=f"Unknown file type '{filetype}'\n"
                                                             f"Choose one from: {', '.join(supported_formats)}")

    def _export_reflections_as_cif(self, phases, histograms, as_intensities=False, pretty=False,
                                   include_histograms=False):
        for phase in phases:
            doc = gemmi.cif.Document()
            cell = self.get_cell(phase)
            spacegroup = self.get_spacegroup(phase)

            # Check if reflections from all histograms are included
            all_histograms = phase.histograms()
            includes_all_histograms = True
            histogram_names = [h.name for h in histograms]
            for histogram_name in all_histograms:
                if histogram_name not in histogram_names:
                    includes_all_histograms = False

            histogram_names = []
            for histogram in histograms:
                histogram_name = histogram.name.split(".")[0][5:].replace("_", "")
                histogram_names.append(histogram_name)

                # Initialize new cif data block
                block = doc.add_new_block(f'r_{histogram_name}_sf')
                block.set_pair('_entry.id', phase.name)

                # Unit-cell
                block.set_pair('_cell.entry_id', phase.name)
                for latt_param, item in zip(cell, ['length_a', 'length_b', 'length_c',
                                                   'angle_alpha', 'angle_beta', 'angle_gamma']):
                    block.set_pair(f'_cell.{item}', str(round(latt_param, 4)))

                # Spacegroup symmetry
                block.set_pair('_symmetry.entry_id', phase.name)
                block.set_pair('_symmetry.space_group_name_H-M', gemmi.cif.quote(spacegroup.hm))
                block.set_pair('_symmetry.Int_Tables_number', str(spacegroup.number))

                # Reflections
                reflections = self.get_reflections(phase, histogram, scale=False)
                if as_intensities:
                    refln_loop = block.init_mmcif_loop('_refln.', ['index_h', 'index_k', 'index_l', 'intensity_meas',
                                                                   'intensity_sigma', 'intensity_calc', 'phase_calc',
                                                                   'd_spacing'])
                else:
                    refln_loop = block.init_mmcif_loop('_refln.', ['index_h', 'index_k', 'index_l', 'F_meas',
                                                                   'F_meas_sigma', 'F_calc', 'phase_calc', 'd_spacing'])

                    # Convert Fo^2, sigma(Fo^2), Fc^2 to Fo, sigma(Fo), Fc
                    reflections[:, [4, 5, 6]] = np.sqrt(reflections[:, [4, 5, 6]])

                hkl = reflections[:, [0, 1, 2]].astype('int')
                sf = reflections[:, [4, 5, 6, 7, 3]].astype('str')
                reflections = np.hstack((hkl, sf)).astype('str').swapaxes(0, 1)
                refln_loop.set_all_values(reflections.tolist())

                # Histogram
                if include_histograms:
                    pdbx_powder_data_loop = block.init_mmcif_loop('_pdbx_powder_data.', ['pd_meas_2theta_scan',
                                                                                         'pd_meas_intensity_total',
                                                                                         'pd_proc_ls_weight',
                                                                                         'pd_calc_intensity_total',
                                                                                         'pd_proc_intensity_bkg_calc'])
                    datapoints = self.get_histogram(histogram).astype('str').swapaxes(0, 1)
                    pdbx_powder_data_loop.set_all_values(datapoints.tolist())

            export_directory = GI.settings.xtl_directories['reflections']
            self._prepare_directory(export_directory)

            # File name: project_phase_bakXX_histograms_I/F_sf.cif
            output_file = GI.settings.working_directory / export_directory / \
                          f'{self._name[:-4]}_{phase.name}_{self._get_gpx_version()}_' \
                          f'{"all" if includes_all_histograms else "-".join(histogram_names)}_' \
                          f'{"I" if as_intensities else "F"}_sf.cif'

            if pretty:
                from xtl.io import mmCIF
                mmCIF(doc).pretty_export(output_file)
            else:
                doc.write_file(output_file, gemmi.cif.Style.Pdbx)
            print(f"Saved reflections for phase '{phase.name}' to {output_file}")
        return

    def _export_reflections_as_mtz(self, phases, histograms, as_intensities=False):

        for phase in phases:
            mtz = gemmi.Mtz()
            mtz.spacegroup = self.get_spacegroup(phase)
            mtz.cell.set(*self.get_cell(phase))

            # Check if reflections from all histograms are included
            all_histograms = phase.histograms()
            includes_all_histograms = True
            histogram_names = [h.name for h in histograms]
            for histogram_name in all_histograms:
                if histogram_name not in histogram_names:
                    includes_all_histograms = False

            raw_reflections = []
            histogram_names = []
            for i, histogram in enumerate(histograms):
                histogram_name = histogram.name.split(".")[0][5:].replace("_", "")
                histogram_names.append(histogram_name)

                # Grab reflections
                new_reflections = self.get_reflections(phase, histogram, scale=True)
                if not as_intensities:
                    # Convert Fo^2, sigma(Fo^2), Fc^2 to Fo, sigma(Fo), Fc
                    new_reflections[:, [4, 5, 6]] = np.sqrt(new_reflections[:, [4, 5, 6]])
                raw_reflections.append(new_reflections)

                dataset = histogram.name.split('.')[0][5:]  # PWDR xxx.gsa
                mtz.add_dataset(dataset)

                # Set dataset metadata
                mtz.datasets[i].project_name = f'{self._name[:-4]}_bak{self._get_gpx_version()}'
                mtz.datasets[i].crystal_name = phase.name
                mtz.datasets[i].wavelength = self.get_wavelength(histogram)[0]

                # Add columns
                if i == 0:
                    mtz.add_column('H', 'H', dataset_id=0)
                    mtz.add_column('K', 'H', dataset_id=0)
                    mtz.add_column('L', 'H', dataset_id=0)
                    if as_intensities:
                        mtz.add_column('I', 'J', dataset_id=0)
                        mtz.add_column('SIGI', 'Q', dataset_id=0)
                        mtz.add_column('IC', 'J', dataset_id=0)
                    else:
                        mtz.add_column('FP', 'F', dataset_id=0)
                        mtz.add_column('SIGFP', 'Q', dataset_id=0)
                        mtz.add_column('FC', 'F', dataset_id=0)
                    mtz.add_column('PHIC', 'P', dataset_id=0)
                else:
                    if as_intensities:
                        mtz.add_column(f'I_{dataset}', 'J', dataset_id=i)  # Fo^2 or Io
                        mtz.add_column(f'SIGI_{dataset}', 'Q', dataset_id=i)  # sigma(Fo^2) or sigma(Io)
                        mtz.add_column(f'IC_{dataset}', 'J', dataset_id=i)  # Fc^2 or Ic
                    else:
                        mtz.add_column(f'FP_{dataset}', 'F', dataset_id=i)  # Fo
                        mtz.add_column(f'SIGFP_{dataset}', 'Q', dataset_id=i)  # sigma(Fo)
                        mtz.add_column(f'FC_{dataset}', 'F', dataset_id=i)  # Fc
                    mtz.add_column(f'PHIC_{dataset}', 'P', dataset_id=i)  # phase

            reflections = self._hstack_reflections(raw_reflections)

            if self.debug:
                columns = mtz.columns
                column_labels = ','.join([c.label for c in columns])
                column_types = ','.join([c.type for c in columns])

                export_directory = GI.settings.xtl_directories['reflections']
                self._prepare_directory(export_directory)
                debug_csv = GI.settings.working_directory / export_directory / \
                            f'{self._name[:-4]}_{phase.name}_bak{self._get_gpx_version()}_' \
                            f'{"all" if includes_all_histograms else "-".join(histogram_names)}_' \
                            f'{"I" if as_intensities else "F"}_debug.csv'
                np.savetxt(debug_csv, reflections, header=f'{column_labels}\n{column_types}', comments='',
                           delimiter=',', fmt='%1.6e')
                print(f'Saved merged reflections to {debug_csv}')

            # Add data to mtz
            mtz.set_data(reflections)

            # Set additional metadata
            from datetime import datetime
            mtz.title = phase.name
            mtz.history = [f"Created with xtl {cfg['xtl']['version'].value} on "
                           f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"]

            export_directory = GI.settings.xtl_directories['reflections']
            self._prepare_directory(export_directory)
            # File name: project_phase_bakXX_histograms_I/F_sf.cif
            output_file = GI.settings.working_directory / export_directory / \
                          f'{self._name[:-4]}_{phase.name}_bak{self._get_gpx_version()}_' \
                          f'{"all" if includes_all_histograms else "-".join(histogram_names)}_' \
                          f'{"I" if as_intensities else "F"}.mtz'

            mtz.write_to_file(output_file)
            print(f"Saved reflections for phase '{phase.name}' to {output_file}")

    @staticmethod
    def _hstack_reflections(reflist):
        """
        Takes a list of reflection arrays (of unequal length) and stacks the structure factors horizontally for each
        array. Used for preparing data before .mtz file export.

        :param list[np.ndarray] or tuple[np.ndarray] reflist: iterable of numpy arrays with the following columns:
            [h, k, l, d, Fo^2, sigma(Fo^2), Fc^2, phase]. Can be prepared by :meth:`.Project.get_reflections`.
        :return: Array with the following columns: [h, k, l, Fo^2_1, sigma(Fo^2)_1, Fc^2_1, phase_1, Fo^2_2, ...]
        :rtype: np.ndarray
        """
        # Get the hkl list from the largest array in the reflist
        largest_array_length, largest_array_index = max((len(array), i) for i, array in enumerate(reflist))
        hkl = reflist[largest_array_index][:, 0:3]

        merged_reflections = hkl
        for array in reflist:
            new_reflections = np.hstack((hkl, np.empty((len(hkl), 4)) * np.nan))
            # new_reflections: h, k, l, NaN, NaN, NaN, NaN
            for i, h in enumerate(hkl):
                try:
                    # assuming that the reflections are in the same order in both arrays
                    if np.array_equal(h, array[i, 0:3]):
                        # new_reflections : h, k, l,    Fo^2, sigma(Fo^2), Fc^2, phase
                        #                   0, 1, 2,    3,    4,           5,    6
                        #
                        # array           : h, k, l, d, Fo^2, sigma(Fo^2), Fc^2, phase
                        #                   0, 1, 2, 3, 4,    5,           6,    7
                        new_reflections[i, 3:7] = array[i, 4:8]
                except IndexError:
                    # Array has less reflections than largest array
                    pass
            # Add the Fo^2, sigma(Fo^2), Fc^2 and phase columns to the merged_reflections array
            merged_reflections = np.hstack((merged_reflections, new_reflections[:, 3:7]))
        return merged_reflections

    def export_map(self, phase, histogram, map_type, grid_step=0.5, omit_map=False, ignore_existing_map=False):
        """
        Export an electron density (Fourier) map for a phase. If a map is already saved in the project, this one will
        exported instead.

        :param GI.G2sc.G2Phase phase:
        :param GI.G2sc.G2PwdrData histogram:
        :param str map_type: One of ``Fo``, ``Fc``, ``Fo-Fc``, ``2Fo-Fc``, ``Fo2``
        :param float grid_step:
        :param bool omit_map: Recalculate phases with OMIT procedure
        :param bool ignore_existing_map: Force map recalculation. Will delete an existing map.
        :return:
        """

        export_directory = GI.settings.xtl_directories['maps']
        self._prepare_directory(export_directory)

        # Check if map is available in .gpx file, else calculate new map
        has_saved_map = False
        if self.has_map(phase) and ignore_existing_map is True:
            print(f"Map already calculated for phase '{phase.name}'. Exporting this instead...")
            output_file = GI.settings.working_directory / export_directory / \
                          f'{self._name[:-4]}_{phase.name}_bak{self._get_gpx_version()}_userCalculated.ccp4'
            has_saved_map = True
        else:
            self.calculate_map(phase, histogram, map_type, grid_step, omit_map)
            histogram_name = histogram.name.split(".")[0][5:].replace("_", "")
            map_type_pretty = GO.get_map_type(map_type).name_pretty
            output_file = GI.settings.working_directory / export_directory / \
                          f'{self._name[:-4]}_{phase.name}_{histogram_name}_{map_type_pretty}' \
                          f'{"_omit" if omit_map else ""}_bak{self._get_gpx_version()}.ccp4'

        # Save map to file
        rho = phase['General']['Map']['rho']
        grid = gemmi.FloatGrid(*rho.shape)
        grid.set_unit_cell(gemmi.UnitCell(*self.get_cell(phase)))
        grid.spacegroup = self.get_spacegroup(phase)
        for (x, y, z), r in np.ndenumerate(rho):
            grid.set_value(x, y, z, r)

        map = gemmi.Ccp4Map()
        map.grid = grid
        map.update_ccp4_header(mode=2, update_stats=True)
        map.write_ccp4_map(output_file)

        if has_saved_map:
            print(f"Saved existing map for phase '{phase.name}' to {output_file}")
        else:
            print(f"Saved {map_type_pretty}{' omit' if omit_map else ''} map for phase '{phase.name}' and histogram "
                  f"'{histogram_name}' to {output_file}")
            self.clear_map(phase)
        return

    def calculate_map(self, phase, histogram, map_type, grid_step=0.5, omit_map=False):
        """
        Calculate an electron density (Fourier) map for a phase. Map data is saved at ``phase['General']['Map']``.

        :param GI.G2sc.G2Phase phase:
        :param GI.G2sc.G2PwdrData histogram:
        :param str map_type: One of ``Fo``, ``Fc``, ``Fo-Fc``, ``2Fo-Fc``, ``Fo2``
        :param float grid_step:
        :param bool omit_map: Recalculate phases with OMIT procedure
        :return:
        """
        self.check_is_phase(phase)
        self.check_is_histogram(histogram)

        map = GO.get_map_type(map_type)
        if omit_map and map.gsas_map_type in ('Fc', 'Fo-Fc', 'Patterson'):
            raise InvalidArgument(raiser='map_type', message=f'{map_type}. {map.gsas_map_type} maps cannot be generated'
                                                             f' with the OMIT procedure. Choose a different map_type '
                                                             f'or set omit_map=False.')

        # Set phase data
        phase['General']['Map']['MapType'] = map.gsas_map_type
        phase['General']['Map']['GridStep'] = grid_step
        reflections = histogram['Reflection Lists'][phase.name]

        # Calculate map
        if omit_map:
            self._calculate_map_omit(phase=phase, reflections=reflections)
        else:
            self._calculate_map_fourier(phase=phase, reflections=reflections)
        return

    @staticmethod
    def _calculate_map_fourier(phase, reflections):
        """
        Calculates a Fourier map for a given reflections list. Space group and map details are extracted from phase
        dict.

        :param GI.G2sc.G2Phase phase:
        :param dict reflections: {'RefList': np.array[h, k, l, multiplicity, d-spacing, 2theta, sig, gam, Fo2, Fc2, ...]
        See: https://gsas-ii.readthedocs.io/en/latest/GSASIIobj.html#powder-reflection-data-structure
        """
        GI.G2sc.SetPrintLevel('none')
        GI.G2m.FourierMap(data=phase, reflDict=reflections)
        GI.G2sc.SetPrintLevel('all')
        return

    @staticmethod
    def _calculate_map_omit(phase, reflections):
        """
        Calculates an omit map for a given reflections list. Space group and map details are extracted from phase
        dict.

        :param GI.G2sc.G2Phase phase:
        :param dict reflections: {'RefList': np.array[h, k, l, multiplicity, d-spacing, 2theta, sig, gam, Fo2, Fc2, ...]
        See: https://gsas-ii.readthedocs.io/en/latest/GSASIIobj.html#powder-reflection-data-structure
        """

        class _DummyProgressBar:
            """
            Used for omit map calculation. Does nothing.
            """

            def Raise(self):
                pass

            def Update(self, nBlk):
                pass

        GI.G2sc.SetPrintLevel('none')
        phase['General']['Map'] = GI.G2m.OmitMap(data=phase, reflDict=reflections, pgbar=_DummyProgressBar())
        GI.G2sc.SetPrintLevel('all')
        return

    def clear_map(self, phase):
        """
        Resets the map dictionary of the phase (i.e. ``phase['General']['Map']``).

        :param GI.G2sc.G2Phase phase:
        :return:
        """
        self.check_is_phase(phase)
        phase['General']['Map'] = GO.MapData().dictionary
        return


class InformationProject(Project):

    def get_filesize(self):
        """
        Returns the filesize of a project in KB, MB etc.

        :return: filesize in appropriate unit
        :rtype: str
        """
        size = GI._path_wrap(self.filename).stat().st_size
        return xm.si_units(value=size, suffix='B', base=1024, digits=2)

    def get_no_of_residue_rigid_bodies(self, phase):
        """
        Returns the number of registered residue rigid bodies in a phase.

        :param GI.G2sc.G2Phase phase:
        :return: no. of residue rigid bodies
        :rtype: int
        """
        self.check_is_phase(phase)
        return len(phase.data['RBModels']['Residue']) if 'Residue' in phase.data['RBModels'] else 0

    def get_no_of_items(self):
        """
        Returns the number of phases, histograms, constraints, restraints and unregistered (unique) rigid bodies in a
        project.

        :return: no_phases, no_histograms, no_constraints, no_restraints, no_rigidbodies
        :rtype: tuple[int, int, int, int, int]
        """
        no_phases = len(self.phases())
        no_histograms = len(self.histograms())

        no_constraints = 0
        for ct in ['Hist', 'HAP', 'Phase', 'Global']:
            constraint_data = self.data['Constraints']['data']
            if ct in constraint_data:
                no_constraints += len(constraint_data[ct])

        no_restraints = 0
        for ph in self.phases():
            restraint_data = self.data['Restraints']['data']
            if ph in restraint_data:
                for rt, r in [('Bond', 'Bonds'), ('Angle', 'Angles'), ('Plane', 'Planes'), ('Chiral', 'Volumes'),
                              ('Torsion', 'Torsions'), ('Rama', 'Ramas'), ('Texture', 'HKLs'), ('ChemComp', 'Sites'),
                              ('General', 'General')]:
                    no_restraints += len(restraint_data[ph.name][rt][r])

        no_rigidbodies = 0
        for rbt in ['Vector', 'Residue']:
            no_rigidbodies += len(self.data['Rigid bodies']['data']['RBIds'][rbt])

        return no_phases, no_histograms, no_constraints, no_restraints, no_rigidbodies


class SimulationProject(Project):

    def simulate_patterns(self):
        max_cycles = self.data['Controls']['data']['max cyc']
        self.data['Controls']['data']['max cyc'] = 0
        self.refine()
        self.data['Controls']['data']['max cyc'] = max_cycles

    def simulate_patterns_sequentially(self):
        raise NotImplemented


class MixtureSimulationProject(SimulationProject):

    def add_simulated_mixture_powder_histogram(self, name, mixture, ttheta_min, ttheta_max, ttheta_step, scale,
                                               iparams):
        # Validate input
        if not isinstance(mixture, PhaseMixture):
            raise InvalidArgument(raiser=f'Simulated histogram {name}', message='Mixture is not type PhaseMixture')
        if not isinstance(iparams, InstrumentalParameters):
            raise InvalidArgument(raiser=f'Simulated histogram {name}', message='IParams is not type '
                                                                                'InstrumentalParameters')

        wavelength = iparams.wavelength
        if isinstance(wavelength, tuple):
            # Get Ka1 for lab iparams
            wavelength = wavelength[0]

        phases = []
        weight_ratios = []
        phase_ratios = []
        for component in mixture.contents:
            # Calculate phase ratios / scale factors for each phase
            phase = component['G2Phase']
            weight_ratio = component['weight_ratio']
            phase_ratio = weight_ratio / phase.data['General']['Mass']
            # Note: For macromolecular phases the resulting phase ratios are too small (e-06),
            #  thus, appear as 0 in the GUI. Should we multiply them by a number?
            phases += [phase]
            weight_ratios += [weight_ratio]
            phase_ratios += [phase_ratio]

            # Check ttheta range for each phase. At least one peak should be included in the range. If not modify range.
            unit_cell = phase.data['General']['Cell'][1:7]
            largest_axis = max(unit_cell[0:3])
            if largest_axis < xm.ttheta_to_d_spacing(ttheta_max, wavelength):
                dmin = largest_axis / 2  # d-spacing uses theta, while ttheta_max is in 2theta
                A = GI.G2lat.cell2A(unit_cell)
                hkld = GI.G2pd.getHKLpeak(dmin=dmin, SGData=phase.data['General']['SGData'], A=A,
                                          Inst=iparams.dictionary)
                new_dmin = hkld[0][3]
                # Add an additional 10% to the required range, to allow part of the first peak to appear
                new_ttheta_max = xm.d_spacing_to_ttheta(dmin, wavelength) * 1.1
                print(f'Adjusting 2theta range to include at least one peak per phase. '
                      f'Was {ttheta_max}, now is {new_ttheta_max}')
                ttheta_max = new_ttheta_max

        # Create simulated histogram
        iparams_file = iparams.save_to_file(name)
        hist = self.add_simulated_powder_histogram(histname=name, phases=phases, Tmin=ttheta_min, Tmax=ttheta_max,
                                                   Tstep=ttheta_step, scale=scale, iparams=iparams_file)
        Path.unlink(iparams_file)

        # Set HAP values (scale factors)
        phase_ratios_sum = sum(phase_ratios)
        for phase, phase_ratio in zip(phases, phase_ratios):
            phase.setHAPvalues({'Scale': [phase_ratio / phase_ratios_sum, False]}, targethistlist=[f'PWDR {name}'])
            # Phase ratios sum is normalized to 1

        return hist
