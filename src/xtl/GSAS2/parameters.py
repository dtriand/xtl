import uuid
from pathlib import Path

import pyxray

from ..GSAS2 import GSAS2Interface as GI
from xtl.exceptions import InvalidArgument, FileError

# Template dictionaries, without wavelength
default_parameters = {
    'synchrotron': {
        'Type': 'PXC',
        'Bank': 1.0,
        'Lam': '',
        'Zero': 0.0,
        'Polariz.': 0.99,
        'Azimuth': 0.0,
        'U': 5.9840407759,
        'V': -1.28771353531,
        'W': 0.118521878603,
        'X': -0.0977791308891,
        'Y': 4.40147397286,
        'Z': 0.0,
        'SH/L': 0.000464356231583
    },
    'lab': {
        'Type': 'PXC',
        'Bank': 1.0,
        'Lam1': '',
        'Lam2': '',
        'I(L2)/I(L1)': '',
        'Zero': 0.0,
        'Polariz.': 0.7,
        'Azimuth': 0.0,
        'U': 2.0,
        'V': -2.0,
        'W': 5.0,
        'X': 0.0,
        'Y': 0.0,
        'Z': 0.0,
        'SH/L': 0.002,
        'Source': ''
    }
}

# Characteristic radiation values from
# Hölzer, G., Fritsch, M., Deutsch, M., Hartwig, J., Forster, E., 1997.
# Kα1,2 and Kβ1,3 x-ray emission lines of the 3d transition metals.
# Physical Review A 56, 4554–4568. https://doi.org/10.1103/PhysRevA.56.4554
default_radiations = {
    '24': {'element': 'Cr', 'Lam1': 2.289726, 'Lam2': 2.293651, 'I(L2)/I(L1)': 0.5},
    '25': {'element': 'Mn', 'Lam1': 2.101854, 'Lam2': 2.105822, 'I(L2)/I(L1)': 0.51},
    '26': {'element': 'Fe', 'Lam1': 1.936041, 'Lam2': 1.939973, 'I(L2)/I(L1)': 0.51},
    '27': {'element': 'Co', 'Lam1': 1.788996, 'Lam2': 1.792835, 'I(L2)/I(L1)': 0.52},
    '28': {'element': 'Ni', 'Lam1': 1.657930, 'Lam2': 1.661756, 'I(L2)/I(L1)': 0.52},
    '29': {'element': 'Cu', 'Lam1': 1.5405929, 'Lam2': 1.5444274, 'I(L2)/I(L1)': 0.52},
    '42': {'element': 'Mo', 'Lam1': 0.70931715, 'Lam2': 0.713607, 'I(L2)/I(L1)': 0.5}
}


class InstrumentalParameters:

    def __init__(self, file):
        """
        A class for manipulating GSAS instrumental parameters files. Both new (.instprm) and old (.prm, .inst, .ins)
        parameter files are supported.

        Alternatively, parameters dictionaries can be generated from user-supplied dictionaries
        (see :meth:`InstrParameters.from_dictionary`) or by template dictionaries (see
        :meth:`InstrParameters.defaults_synchrotron`, :meth:`InstrParameters.defaults_lab`).

        The object contains the following class variables:
            *   :meth:`InstrParameters.dictionary`: contains a dictionary of all instrumental parameters, that is also
                compatible with GSAS2. Each entry is in the form ``[initial_value, modified_value, refinement_status]``
            *   :meth:`InstrParameters.type`: Data type as defined by GSAS
            *   :meth:`InstrParameters.bank`: Bank number
            *   :meth:`InstrParameters.wavelength`: Wavelength in Angstroms. One value for synchrotron radiation or
                tuple of three values for laboratory radiation ``(Ka1, Ka2, I(Ka2)/I(Ka1))``
            *   :meth:`InstrParameters.source`: X-ray source type. ``None`` for synchrotron, tube element for laboratory
                source
            *   :meth:`InstrParameters.zero_shift`: Instrumental angular offset due to misalignment in degrees 2theta
            *   :meth:`InstrParameters.polarization`: X-ray polarization factor
            *   :meth:`InstrParameters.azimuth`: Azimuthal angle for texture analysis
            *   :meth:`InstrParameters.gaussian_U`: Peak shape parameter U
            *   :meth:`InstrParameters.gaussian_V`: Peak shape parameter V
            *   :meth:`InstrParameters.gaussian_W`: Peak shape parameter W
            *   :meth:`InstrParameters.lorentzian_X`: Peak shape parameter X
            *   :meth:`InstrParameters.lorentzian_Y`: Peak shape parameter Y
            *   :meth:`InstrParameters.lorentzian_Z`: Peak shape parameter Z
            *   :meth:`InstrParameters.axial_divergence`: Peak shape parameter SH/L

        :param str file: filename
        """

        # Load a lightweight reader
        powder_reader = GI.G2fil.LoadImportRoutines("pwd_csv", "Powder_Data")[0]

        # Parses both new (.instprm) and old (.prm, .inst, .ins) parameters files
        self._initial_dict = GI.G2sc.load_iprms(instfile=file, reader=powder_reader)[0]

        # Check that all keys are available in the dictionary
        required_keys = ['Type', 'Bank', 'Zero', 'Polariz.', 'Azimuth', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SH/L']
        for key in required_keys:
            if key not in self._initial_dict:
                raise FileError(file=file, message=f"Could not find key '{key}'")

        # Check that either a 'Lam' or all three 'Lam1', 'Lam2' and 'I(L2)/I(L1)' entries are present
        if 'Lam' not in self._initial_dict and ('Lam1' not in self._initial_dict or
                                                'Lam2' not in self._initial_dict or
                                                'I(L2)/I(L1)' not in self._initial_dict):  # False if no wavelength
            raise FileError(file=file, message='Could not find wavelength',
                            details="Either 'Lam' or all three 'Lam1', 'Lam2', 'I(L2)/I(L1)' need to be specified.")

        # Save initial values as private attributes
        self._type = self._initial_dict['Type'][0]
        self._bank = self._initial_dict['Bank'][0]

        if 'Lam' in self._initial_dict:
            # For synchrotron data
            self._wavelength = self._initial_dict['Lam'][0]
        elif ('Lam1' and 'Lam2' and 'I(L2)/I(L1)') in self._initial_dict:
            # For lab data
            self._wavelength1 = self._initial_dict['Lam1'][0]
            self._wavelength2 = self._initial_dict['Lam2'][0]
            self._wavelength_ratio = self._initial_dict['I(L2)/I(L1)'][0]
            if 'Source' in self._initial_dict:
                self._source = self._initial_dict['Source'][0]

        self._zero_shift = self._initial_dict['Zero'][0]
        self._polarization = self._initial_dict['Polariz.'][0]
        self._azimuth = self._initial_dict['Azimuth'][0]
        self._gaussian_U = self._initial_dict['U'][0]
        self._gaussian_V = self._initial_dict['V'][0]
        self._gaussian_W = self._initial_dict['W'][0]
        self._lorentzian_X = self._initial_dict['X'][0]
        self._lorentzian_Y = self._initial_dict['Y'][0]
        self._lorentzian_Z = self._initial_dict['Z'][0]
        self._axial_divergence = self._initial_dict['SH/L'][0]

    @property
    def type(self):
        """
        Data type as defined by GSAS
        """
        return self._type

    @type.setter
    def type(self, new_type):
        if isinstance(new_type, str):  # additional checks might be needed
            self._type = new_type
        else:
            raise InvalidArgument(raiser='type', message=f'{new_type}. Must be str')

    @property
    def bank(self):
        """
        Bank number in parameters file
        """
        return self._bank

    @bank.setter
    def bank(self, new_bank):
        if isinstance(new_bank, (float, int)):
            self._bank = float(new_bank)
        else:
            raise InvalidArgument(raiser='bank', message=f'{new_bank}. Must be number.')

    @property
    def wavelength(self):
        """
        Wavelength in Angstroms. A single value for synchrotron radiation, or a tuple of three values for laboratory
        radiation ``(Ka1, Ka2, I(Ka2)/I(Ka1))``
        """
        if hasattr(self, '_wavelength'):
            # For synchrotron data
            return self._wavelength
        elif hasattr(self, '_wavelength1') and hasattr(self, '_wavelength2') and hasattr(self, '_wavelength_ratio'):
            # For laboratory data
            return self._wavelength1, self._wavelength2, self._wavelength_ratio

    @wavelength.setter
    def wavelength(self, new_wavelength):
        if hasattr(self, '_wavelength'):
            # For synchrotron data
            if isinstance(new_wavelength, (float, int)) and new_wavelength > 0:
                self._wavelength = float(new_wavelength)
            else:
                raise InvalidArgument(raiser='wavelength', message=f'{new_wavelength}. Must be positive number.')
        elif hasattr(self, '_wavelength1') and hasattr(self, '_wavelength2') and hasattr(self, '_wavelength_ratio'):
            # For laboratory data
            if isinstance(new_wavelength, (list, tuple)) and len(new_wavelength) == 3:
                l1, l2, r = new_wavelength
                if (isinstance(l1, (float, int)) and l1 > 0) and \
                        (isinstance(l2, (float, int)) and l2 > 0) and \
                        (isinstance(r, (float, int)) and r > 0):
                    self._wavelength1 = l1
                    self._wavelength2 = l2
                    self._wavelength_ratio = r
                    return

            raise InvalidArgument(raiser='wavelength', message=f'{new_wavelength}. Must be iterable of three positive '
                                                               f'numbers: (Wavelength1, Wavelength2, IntensityRatio)')

    @property
    def source(self):
        """
        Radiation source, only for characteristic radiations
        """
        if hasattr(self, '_source'):
            return self._source
        return None

    @property
    def zero_shift(self):
        """
        Angular offset of instrument from 0
        """
        return self._zero_shift

    @zero_shift.setter
    def zero_shift(self, new_zero):
        if isinstance(new_zero, (float, int)):
            self._zero_shift = float(new_zero)
        else:
            raise InvalidArgument(raiser='zero shift', message=f'{new_zero}. Must be float or int.')

    @property
    def polarization(self):
        """
        Polarization factor for x-ray source
        """
        return self._polarization

    @polarization.setter
    def polarization(self, new_polar):
        if isinstance(new_polar, (float, int)) and 0.5 <= new_polar <= 1:
            self._polarization = new_polar
        else:
            raise InvalidArgument(raiser='polarization factor', message=f'{new_polar}. Must be a number between 0.5 '
                                                                        f'and 1.0')

    @property
    def azimuth(self):
        """
        Azimuthal angle for texture analysis
        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, new_azimuth):
        if isinstance(new_azimuth, (float, int)):
            self._azimuth = new_azimuth
        else:
            raise InvalidArgument(raiser='azimuth', message=f'{new_azimuth}. Must be float or int.')

    @property
    def gaussian_U(self):
        """
        Peak shape parameter U for Gaussian peak component
        """
        return self._gaussian_U

    @gaussian_U.setter
    def gaussian_U(self, new_U):
        if isinstance(new_U, (float, int)):
            self._gaussian_U = float(new_U)
        else:
            raise InvalidArgument(raiser='gaussian U', message=f'{new_U}. Must be float or int.')

    @property
    def gaussian_V(self):
        """
        Peak shape parameter V for Gaussian peak component
        """
        return self._gaussian_V

    @gaussian_V.setter
    def gaussian_V(self, new_V):
        if isinstance(new_V, (float, int)):
            self._gaussian_V = float(new_V)
        else:
            raise InvalidArgument(raiser='gaussian V', message=f'{new_V}. Must be float or int.')

    @property
    def gaussian_W(self):
        """
        Peak shape parameter W for Gaussian peak component
        """
        return self._gaussian_W

    @gaussian_W.setter
    def gaussian_W(self, new_W):
        if isinstance(new_W, (float, int)):
            self._gaussian_W = float(new_W)
        else:
            raise InvalidArgument(raiser='gaussian W', message=f'{new_W}. Must be float or int.')

    @property
    def lorentzian_X(self):
        """
        Peak shape parameter X for Lorentzian peak component
        """
        return self._lorentzian_X

    @lorentzian_X.setter
    def lorentzian_X(self, new_X):
        if isinstance(new_X, (float, int)):
            self._lorentzian_X = float(new_X)
        else:
            raise InvalidArgument(raiser='lorentzian X', message=f'{new_X}. Must be float or int.')

    @property
    def lorentzian_Y(self):
        """
        Peak shape parameter Y for Lorentzian peak component
        """
        return self._lorentzian_Y

    @lorentzian_Y.setter
    def lorentzian_Y(self, new_Y):
        if isinstance(new_Y, (float, int)):
            self._lorentzian_Y = float(new_Y)
        else:
            raise InvalidArgument(raiser='lorentzian Y', message=f'{new_Y}. Must be float or int.')

    @property
    def lorentzian_Z(self):
        """
        Peak shape parameter Z for Lorentzian peak component
        """
        return self._lorentzian_Z

    @lorentzian_Z.setter
    def lorentzian_Z(self, new_Z):
        if isinstance(new_Z, (float, int)):
            self._lorentzian_Z = float(new_Z)
        else:
            raise InvalidArgument(raiser='lorentzian Z', message=f'{new_Z}. Must be float or int.')

    @property
    def axial_divergence(self):
        """
        Peak shape parameter SH/L for peak asymmetry resulting from axial divergence.
        """
        return self._axial_divergence

    @axial_divergence.setter
    def axial_divergence(self, new_shl):
        if isinstance(new_shl, (float, int)):
            self._axial_divergence = float(new_shl)
        else:
            raise InvalidArgument(raiser='SH/L', message=f'{new_shl}, Must be float or int.')

    @property
    def dictionary(self):

        # key = [original_value, modified_value, refinement_status]
        output = {}

        output['Type'] = [self._initial_dict['Type'][0], self._type, False]
        output['Bank'] = [self._initial_dict['Bank'][0], self._bank, False]

        if hasattr(self, '_wavelength'):
            output['Lam'] = [self._initial_dict['Lam'][0], self._wavelength, False]
        elif hasattr(self, '_wavelength1') and hasattr(self, '_wavelength2') and hasattr(self, '_wavelength_ratio'):
            output['Lam1'] = [self._initial_dict['Lam1'][0], self._wavelength1, False]
            output['Lam2'] = [self._initial_dict['Lam2'][0], self._wavelength2, False]
            output['I(L2)/I(L1)'] = [self._initial_dict['I(L2)/I(L1)'][0], self._wavelength_ratio, False]

        output['Zero'] = [self._initial_dict['Zero'][0], self._zero_shift, False]
        output['Polariz.'] = [self._initial_dict['Polariz.'][0], self._polarization, False]
        output['Azimuth'] = [self._initial_dict['Azimuth'][0], self._azimuth, False]
        output['U'] = [self._initial_dict['U'][0], self._gaussian_U, False]
        output['V'] = [self._initial_dict['V'][0], self._gaussian_V, False]
        output['W'] = [self._initial_dict['W'][0], self._gaussian_W, False]
        output['X'] = [self._initial_dict['X'][0], self._lorentzian_X, False]
        output['Y'] = [self._initial_dict['Y'][0], self._lorentzian_Y, False]
        output['Z'] = [self._initial_dict['Z'][0], self._lorentzian_Z, False]
        output['SH/L'] = [self._initial_dict['SH/L'][0], self._axial_divergence, False]

        if hasattr(self, '_source'):
            output['Source'] = [self._initial_dict['Source'][0], self._source, False]

        return output

    @dictionary.setter
    def dictionary(self, new_dict: dict):
        if not isinstance(new_dict, dict):
            return

        # Check that either a 'Lam' or all three 'Lam1', 'Lam2' and 'I(L2)/I(L1)' entries are present
        if 'Lam' not in new_dict.keys() and ('Lam1' not in new_dict.keys() or
                                             'Lam2' not in new_dict.keys() or
                                             'I(L2)/I(L1)' not in new_dict.keys()):
            raise InvalidArgument(raiser='dictionary', message="Could not find wavelength in the provided dictionary. "
                                                               "Either 'Lam' or all three 'Lam1', 'Lam2', 'I(L2)/I(L1)'"
                                                               " need to be specified.")

        required_keys = ['Type', 'Bank', 'Zero', 'Polariz.', 'Azimuth', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SH/L']
        for key in required_keys:
            # Check whether all required keys are present
            if key not in new_dict.keys():
                raise InvalidArgument(raiser='dictionary', message=f"Could not find key '{key}' in the provided "
                                                                   f"dictionary.")

        wavelength = {}  # for lab data wavelength assignment
        required_keys += ['Lam', 'Lam1', 'Lam2', 'I(L2)/I(L1)']
        for key, value in new_dict.items():
            if key not in required_keys:
                # Ignore any non-standard keys
                continue
            elif key == 'Type':
                self.type = value
            elif key == 'Bank':
                self.bank = value
            elif key == 'Lam':
                self.wavelength = value
            elif key == 'Lam1':
                wavelength['Lam1'] = value
            elif key == 'Lam2':
                wavelength['Lam2'] = value
            elif key == 'I(L2)/I(L1)':
                wavelength['I(L2)/I(L1)'] = value
            elif key == 'Zero':
                self.zero_shift = value
            elif key == 'Polariz.':
                self.polarization = value
            elif key == 'Azimuth':
                self.azimuth = value
            elif key == 'U':
                self.gaussian_U = value
            elif key == 'V':
                self.gaussian_V = value
            elif key == 'W':
                self.gaussian_W = value
            elif key == 'X':
                self.lorentzian_X = value
            elif key == 'Y':
                self.lorentzian_Y = value
            elif key == 'Z':
                self.lorentzian_Z = value
            elif key == 'SH/L':
                self.axial_divergence = value

        # Lab data wavelength assignment
        if wavelength:
            self.wavelength = wavelength['Lam1'], wavelength['Lam2'], wavelength['I(L2)/I(L1)']

    def reset(self):
        """
        Dump all modified values and restore the initial values to the dictionary.

        :return:
        """
        new_dict = {}
        for key, value in self.dictionary.items():
            new_dict[key] = value[0]
        self.dictionary = new_dict

    def save_to_file(self, name):
        """
        Save instrumental parameters to .instprm file. Any values that have previously been modified will be saved to
        file.

        :param str name: filename
        :return: filename
        """
        export_dict = {}
        for key, value in self.dictionary.items():
            export_dict[key] = value[1]  # save modified values to file
        return self._dict_to_instprm(export_dict, GI._path_wrap(name))

    @staticmethod
    def _dict_to_instprm(dictionary, name):
        """
        Save a dictionary to a .instprm-like file. All dictionary entries are saved; no validation is performed.

        :param dict dictionary: dictionary to convert
        :param str name: filename
        :return: filename
        """
        if isinstance(name, str):
            if name.endswith('.instprm'):
                # Remove extension if it's already provided in filename
                name = name[:-8]
        elif isinstance(name, Path):
            if name.suffix == '.instprm':
                name = name.name
        with open(f'{name}.instprm', 'w') as fp:
            fp.write('#GSAS-II instrument parameter file\n')  # ensures readability by GSAS2 readers
            for key, value in dictionary.items():
                fp.write(f'{key}:{value}\n')  # 'key:value'
        return f'{name}.instprm'

    @classmethod
    def from_dictionary(cls, dictionary):
        """
        Create instrumental parameters from a user-defined dictionary.

        :param dict dictionary:
        :return cls:
        """
        # Create a temporary .instprm file to be parsed by the GSAS2 reader
        temporary_file = GI._path_wrap(uuid.uuid4().hex)
        cls._dict_to_instprm(dictionary, f'{temporary_file}')
        child = cls(file=f'{temporary_file}.instprm')
        Path(f'{temporary_file}.instprm').unlink()
        return child

    @classmethod
    def defaults_synchrotron(cls, wavelength=1.0):
        """
        Generate instrumental parameters with typical values for a synchrotron diffractometer. The wavelength can be
        specified with the ``wavelength`` argument (defaults to 1).

        :param int or float wavelength: wavelength in Angstroms
        :return cls:
        """
        if not isinstance(wavelength, (int, float)) or wavelength <= 0:
            raise InvalidArgument(f'{wavelength}. Must be a positive number.', 'wavelength')
        synchr_dict = default_parameters['synchrotron']
        synchr_dict['Lam'] = wavelength
        return cls.from_dictionary(synchr_dict)

    @classmethod
    def defaults_lab(cls, tube='Cu'):
        """
        Generate instrumental parameters with typical values for a laboratory diffractometer. The tube element can be
        specified with the ``tube`` argument (defaults to copper).

        Ka1, Ka2 and I(Ka2)/I(Ka1) values can be generated for every element. If the element is one of Cr, Mn, Fe, Co,
        Ni, Cu, Mo, then the values are tabulated from 'International Tables Vol C.', else approximate values are
        calculated.

        :param str or int tube: element symbol, name or atomic number
        :return cls:
        """
        if not isinstance(tube, (str, int)):
            raise InvalidArgument(raiser='tube', message=f'{tube}. Tube must be str or int.')
        try:
            if isinstance(tube, str):
                # Correct possible typos e.g. 'ag' -> 'Ag'
                # and also enable element names to be parsed
                # (pyxray understands 'Copper', but not 'copper')
                tube = tube.capitalize()

            # Interpret element with pyxray
            elem = pyxray.element(tube)
        except pyxray.NotFound:
            raise InvalidArgument(raiser='tube',
                                  message=f"Unknown {'element' if isinstance(tube, str) else 'atomic number'} '{tube}'")

        l1, l2, r, s = 0, 0, 0, ''
        if str(elem.atomic_number) in default_radiations.keys():
            # Some of the most used tube elements are tabulated, both for speed and accuracy
            # Use tabulated values when possible
            radiation = default_radiations[str(elem.atomic_number)]
            l1, l2, r = radiation['Lam1'], radiation['Lam2'], radiation['I(L2)/I(L1)']
            s = f"{radiation['element']}Ka (tabulated)"
        else:
            line1 = pyxray.xray_line(elem, 'Ka1')
            line2 = pyxray.xray_line(elem, 'Ka2')
            # Check that all required values are available for calculations
            if line1.energy_eV and line2.energy_eV and line2.relative_weight:
                # Convert photon energies to wavelengths, using E(eV)=12400/lambda(A)
                l1 = round(12398.41857 / line1.energy_eV, 3)  # Truncate values to avoid disagreement of pyxray values
                l2 = round(12398.41857 / line2.energy_eV, 3)  # with literature values
                r = round(line2.relative_weight, 1)  # I(Ka2)/I(Ka1)
                s = f"{pyxray.element_symbol(elem)}Ka (computed)"

        if l1 and l2 and r:
            # Generate instrumental parameters only if all values calculated
            lab_dict = default_parameters['lab']
            lab_dict['Lam1'] = l1
            lab_dict['Lam2'] = l2
            lab_dict['I(L2)/I(L1)'] = r
            lab_dict['Source'] = s
            return cls.from_dictionary(lab_dict)
        else:
            # Usually for atomic numbers larger than 100
            raise InvalidArgument(raiser='tube', message=f"Characteristic radiation could not be computed for "
                                                         f"element '{s}'")
