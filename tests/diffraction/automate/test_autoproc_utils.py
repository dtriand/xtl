import pytest

from pathlib import Path

from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig


class TestAutoPROCConfig:

    params = ['unit_cell', 'space_group', 'wavelength', 'resolution_low', 'resolution_high', 'anomalous',
              'no_residues', 'mosaicity', 'rfree_mtz', 'reference_mtz', 'mtz_project_name', 'mtz_crystal_name',
              'mtz_dataset_name', 'xds_njobs', 'xds_nproc', 'xds_lib', 'xds_polarization_fraction',
              'xds_defpix_optimize', 'xds_idxref_optimize', 'xds_n_background_images',
              '_XdsExcludeIceRingsAutomatically', '_RunIdxrefExcludeIceRingShells', 'exclude_ice_rings',
              'beamline', 'resolution_cuttoff_criterion', 'batch_mode']
    list_params = ['xds_idxref_refine_params', 'xds_integrate_refine_params', 'xds_correct_refine_params',
                   '_macros', '_args']
    dict_params = ['extra_params']

    def test_defaults(self):
        config = AutoPROCConfig()

        for param in self.params:
            assert getattr(config, param) is None

        for param in self.list_params:
            assert getattr(config, param) == []

        for param in self.dict_params:
            assert getattr(config, param) == {}


    def test_formatters(self):
        config = AutoPROCConfig(
            unit_cell=[78, 78, 37, 90, 90, 90],
            space_group='P 43 21 2',
            resolution_low=38.0,
            resolution_high=1.2,
            rfree_mtz=Path('path/to/rfree.mtz'),
            xds_idxref_refine_params=['POSITION', 'BEAM', 'AXIS'],
            xds_integrate_refine_params=['POSITION', 'BEAM', 'AXIS'],
            xds_correct_refine_params=['POSITION', 'BEAM', 'AXIS'],
            exclude_ice_rings=True,
            extra_params={'test': 'value'}
        )
        assert config.get_all_params(modified_only=True, grouped=True) == {
            'user_params': {
                'comment': 'User parameters',
                'params': {
                    'cell': '"78.0 78.0 37.0 90.0 90.0 90.0"',
                    'symm': 'P43212',
                    'init_reso': '"38.00 1.20"',
                    'free_mtz': '"path/to/rfree.mtz"'
                }
            },
            'ice_rings_params': {
                'comment': 'Ice rings parameters',
                'params': {
                    'RunIdxrefExcludeIceRingShells': 'yes',
                    'XdsExcludeIceRingsAutomatically': 'yes'
                }
            },
            'xds_params': {
                'comment': 'XDS parameters',
                'params': {
                    'autoPROC_XdsKeyword_REFINEIDXREF': '"POSITION BEAM AXIS"',
                    'autoPROC_XdsKeyword_REFINEINTEGRATE': '"POSITION BEAM AXIS"',
                    'autoPROC_XdsKeyword_REFINECORRECT': '"POSITION BEAM AXIS"'
                }
            },
        }
        assert config.get_group('extra_params') == {'_extra_params': {'test': 'value'}}

        assert config.get_all_params(modified_only=True, grouped=False) == {
            'cell': '"78.0 78.0 37.0 90.0 90.0 90.0"',
            'symm': 'P43212',
            'init_reso': '"38.00 1.20"',
            'free_mtz': '"path/to/rfree.mtz"',
            'RunIdxrefExcludeIceRingShells': 'yes',
            'XdsExcludeIceRingsAutomatically': 'yes',
            'autoPROC_XdsKeyword_REFINEIDXREF': '"POSITION BEAM AXIS"',
            'autoPROC_XdsKeyword_REFINEINTEGRATE': '"POSITION BEAM AXIS"',
            'autoPROC_XdsKeyword_REFINECORRECT': '"POSITION BEAM AXIS"',
            '_extra_params': {
                'test': 'value'
            }
        }

    def test_macros(self):
        config = AutoPROCConfig(beamline='PetraIIIP14', resolution_cuttoff_criterion='CC1/2')
        assert config.get_param_value('_macros') == {
            '_macros': '-M PetraIIIP14 -M HighResCutOnCChalf'
        }

    def test_args(self):
        config = AutoPROCConfig()
        assert config.get_param_value('_args') == {
            '__args': ''
        }

        config = AutoPROCConfig(beamline='PetraIIIP14', resolution_cuttoff_criterion='CC1/2', batch_mode=True)
        assert config.get_param_value('_args') == {
            '__args': '-B -M PetraIIIP14 -M HighResCutOnCChalf'
        }

    def test_get_all_params(self):
        config = AutoPROCConfig()
        params = config.get_all_params(modified_only=True)
        assert params == {}