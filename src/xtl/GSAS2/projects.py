import xtl.GSAS2.GSAS2Interface as GI
import xtl.math as xm
from xtl.GSAS2.components import PhaseMixture
from xtl.GSAS2.parameters import InstrumentalParameters
from xtl.exceptions import InvalidArgument, FileError
from xtl import cfg

import os
import gemmi
import numpy as np


class Project(GI.G2sc.G2Project):

    def __init__(self, filename, debug=False):
        self.debug = debug
        if os.path.exists(GI._path_wrap(filename)):
            super().__init__(gpxfile=GI._path_wrap(filename))
        else:
            super().__init__(newgpx=GI._path_wrap(filename))
        self._directory, self._name = os.path.split(self.filename)

    def _backup_gpx(self):
        backup_dir = os.path.join(GI.working_directory, '.xtl')
        backup_gpx = os.path.join(backup_dir, self._name)
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        import shutil
        shutil.copy2(src=self.filename, dst=backup_gpx)
        if self.debug:
            print(f'Backing up .gpx file at {backup_gpx}')

    def _get_gpx_version(self):
        """
        Finds the last project.bakXX.gpx file in the folder and returns XX + 1. If no .bak.gpx file is found in the
        directory, returns 0. Can find files up to .bak999.gpx

        :return:
        """
        filename = os.path.splitext(self._name)[0]
        from glob import glob
        bak1 = glob(f'{self._directory}/{filename}.bak?.gpx')
        bak2 = glob(f'{self._directory}/{filename}.bak??.gpx')
        bak3 = glob(f'{self._directory}/{filename}.bak???.gpx')
        if bak3:
            bak3.sort()
            last_file = bak3[-1]
        else:
            if bak2:
                bak2.sort()
                last_file = bak2[-1]
            else:
                if bak1:
                    bak1.sort()
                    last_file = bak1[-1]
                else:
                    # last_file = f'{self._directory}/{filename}.bak-1.gpx'
                    return 0
        file_version = os.path.splitext(last_file)[0].split('.bak')[1]  # Get the number after .bak
        file_version = int(file_version) + 1
        return file_version

    def add_comment(self, comment):
        from datetime import datetime
        xtl_vers = cfg['xtl']['version'].value
        now = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
        self.data['Notebook']['data'].append(f"xtl {xtl_vers} @ {now}\n{comment}")

    def add_phase(self, file, name=None, histograms=[], type=''):
        if not name:
            name = os.path.splitext(os.path.split(file)[1])[0]  # grab filename from a full path
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

    def get_spacegroup(self, phase):
        """
        Returns a Gemmi representation of a phase's spacegroup.

        :param GI.G2sc.G2Phase phase:
        :return: Gemmi spacegroup object
        :rtype: gemmi.SpaceGroup
        """
        self.check_is_phase(phase)
        return gemmi.find_spacegroup_by_name(hm=phase.data['General']['SGData']['SpGrp'])

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


class SimulationProject(Project):

    def simulate_patterns(self):
        max_cycles = self.data['Controls']['data']['max cyc']
        self.data['Controls']['data']['max cyc'] = 0
        self.refine()
        self.data['Controls']['data']['max cyc'] = max_cycles

    def simulate_patterns_sequentially(self):
        pass


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

            # Check ttheta range for each phase. At least one peak should be inluded in the range. If not modify range.
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
                print(f'Adjuxting 2theta range to include at least one peak per phase. '
                      f'Was {ttheta_max}, now is {new_ttheta_max}')
                ttheta_max = new_ttheta_max

        # Create simulated histogram
        iparams_file = iparams.save_to_file(name)
        hist = self.add_simulated_powder_histogram(histname=name, phases=phases, Tmin=ttheta_min, Tmax=ttheta_max,
                                                   Tstep=ttheta_step, scale=scale, iparams=iparams_file)
        os.remove(iparams_file)

        # Set HAP values (scale factors)
        phase_ratios_sum = sum(phase_ratios)
        for phase, phase_ratio in zip(phases, phase_ratios):
            phase.setHAPvalues({'Scale': [phase_ratio / phase_ratios_sum, False]}, targethistlist=[f'PWDR {name}'])
            # Phase ratios sum is normalized to 1

        return hist


class ExportProject(Project):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backup_gpx()

    @staticmethod
    def get_cell(phase):
        if not isinstance(phase, GI.G2sc.G2Phase):
            raise
        return tuple(phase.get_cell().values())[:-1]

    @staticmethod
    def get_wavelength(histogram):
        if not isinstance(histogram, GI.G2sc.G2PwdrData):
            raise
        iparams = histogram.InstrumentParameters
        if 'Lam' in iparams:
            return iparams['Lam'][1]
        elif 'Lam1' in iparams:
            return iparams['Lam1'][1]

    @staticmethod
    def get_histogram(histogram, subtract_background=False):
        """
        Returns histogram datapoints as numpy array. The following columns are included: [2theta, Io, sigma(Io), Ic,
        background]. If ``subtract_background=True``, the returned array has the following 4 columns instead [2theta,
        Io-background, sigma(Io), Ic-background].

        :param histogram:
        :param subtract_background:
        :return:
        """
        if not isinstance(histogram, GI.G2sc.G2PwdrData):
            raise
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
