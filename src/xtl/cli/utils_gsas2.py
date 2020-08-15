from xtl.GSAS2 import GSAS2Interface as GI
from xtl.math import round_value_and_esd, si_units


alpha = '\u03b1'
beta = '\u03b2'
gamma = '\u03b3'
delta = '\u03b4'
theta = '\u03b8'
angstrom = '\u212b'
degree = '\u00b0'
superscript_three = '\u00b3'


def get_phase_info(gpx, phase, verbose=0):
    """

    :param xtl.GSAS2.projects.InformationProject gpx:
    :param GI.G2sc.G2Phase phase:
    :param verbose:
    :return:
    """
    info = {
        'ID': phase.id,
        'Name': phase.name,
        'Space Group': f'{gpx.get_spacegroup(phase).hm} ({gpx.get_spacegroup(phase).number})',
        'Type': gpx.get_phase_type(phase).title()
    }

    if verbose:
        cell = tuple(phase.get_cell().values())
        lps = cell[0:6]
        volume = cell[6]
        info[f'Lattice parameters ({angstrom},{degree})'] = ', '.join([str(round(lp, 3)) for lp in lps])
        info[f'Volume ({angstrom}{superscript_three})'] = round(volume, 3)

    info['Composition'] = gpx.get_formula(phase)
    info['Histograms'] = len(phase.histograms())

    if verbose:
        info['Rigid Bodies'] = gpx.get_no_of_residue_rigid_bodies(phase)
        info['Map'] = 'Yes' if gpx.has_map(phase) else 'No'
        info['RanID'] = phase.ranId

    return info


def get_histogram_info(gpx, histogram, verbose=0):
    """

    :param xtl.GSAS2.projects.Project gpx:
    :param GI.G2sc.G2PwdrData histogram:
    :param verbose:
    :return:
    """
    info = {
        'ID': histogram.id,
        'Name': histogram.name,
        f'2{theta} range ({degree})': ' - '.join([str(round(ttheta, 4)) for ttheta in gpx.get_data_range(histogram)])
    }

    if verbose:
        wavelength, is_lab = gpx.get_wavelength(histogram)
        info[f'Wavelength ({angstrom})'] = f'{wavelength} {"(L1)" if is_lab else ""}'
        info['RanID'] = histogram.ranId

    return info


def get_cell_info(phase, verbose=0):
    """

    :param GI.G2sc.G2Phase phase:
    :param verbose:
    :return:
    """
    lattice_params = ['a', 'b', 'c', alpha, beta, gamma, 'V']
    lattice_params_units = [angstrom] * 3 + [degree] * 3 + [f'{angstrom}{superscript_three}']
    cell = tuple(phase.get_cell().values())
    if verbose:
        cell_esd = tuple(phase.get_cell_and_esd()[1].values())
    else:
        cell_esd = [0] * 7

    info = {}
    for cv, ce, lp, lpu in zip(cell, cell_esd, lattice_params, lattice_params_units):
        if ce == 0:
            # a (A) : 78.4124
            info[f'{lp} ({lpu})'] = cv
        else:
            # a(da) [A] : 78.41(3)
            v, e, s = round_value_and_esd(cv, ce)
            if s > 0:  # esd is < 1 (e.g. 0.13)
                info[f'{lp}({delta}{lp}) [{lpu}]'] = f'{v}({int(e * (10 ** s))})'
            else:
                info[f'{lp}({delta}{lp}) [{lpu}]'] = f'{v:.0f}({int(e)})'

    return info


def get_density_info(phase, verbose=0):
    """

    :param GI.G2sc.G2Phase phase:
    :param verbose:
    :return:
    """
    mass = GI.G2m.getMass(phase.data['General'])
    density, matthews_coeff = GI.G2m.getDensity(phase.data['General'])
    v_solvent = 1 - (1.230 / matthews_coeff)
    v_protein = 1 - v_solvent

    info = {
        'Mass (Da)': round(mass, 4),
        f'Density (Da/{angstrom}{superscript_three})': round(density, 4),
        f'Matthews coef. ({angstrom}{superscript_three}/Da)': round(matthews_coeff, 4)
    }

    if verbose:
        info['Solvent content (%)'] = round(v_solvent * 100, 3)
        info['Protein content (%)'] = round(v_protein * 100, 3)
    return info


def get_map_info(phase, verbose=0):
    """

    :param GI.G2sc.G2Phase phase:
    :param verbose:
    :return:
    """
    map_data = phase['General']['Map']
    gsas_map_type = map_data['MapType']
    map_types = {
        'Fobs': 'Fo',
        'Fcalc': 'Fc',
        'delt-F': 'Fo-Fc',
        '2*Fo-Fc': '2Fo-Fc',
        'Omit': 'Omit (Fo)',
        '2Fo-Fc Omit': 'Omit (2Fo-Fc)',
        'Patterson': 'Patterson'
    }

    info = {
        'Type': map_types[gsas_map_type],
        'Histogram': ', '.join(map_data['RefList']),
        'Grid step': map_data['GridStep']
    }

    if verbose:
        # info['Map size'] = map_data['mapSize'],
        info['Maximum density'] = round(map_data['rhoMax'], 3)
        info['Map size'] = si_units(map_data['rho'].nbytes, suffix='B', base=1024, digits=2)
    return info
