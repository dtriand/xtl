import pytest

from pathlib import Path

from tests.conftest import seed
from xtl.automate.shells import BashShell
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.diffraction.automate.autoproc import AutoPROCJob
from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig


no_datasets = 2
no_images = 100
cbf_images = [f'./a/b/c/dataset_1_{j+1}_{i+1:04d}.cbf' for j in range(no_datasets)  for i in range(no_images)]
h5_images = [f'./a/b/c/dataset_1_{j+1}_master.h5' for j in range(no_datasets)] + \
            [f'./a/b/c/dataset_1_{j+1}_data_{i+1:04d}.h5' for j in range(no_datasets)  for i in range(no_images)]


@pytest.fixture
def datasets(temp_files) -> list[DiffractionDataset]:
    d = DiffractionDataset.from_image(temp_files[0])
    p = d.raw_data_dir / 'processed'
    if d._is_h5:
        return [DiffractionDataset.from_image(temp_files[k], processed_data_dir=p) for k in range(no_datasets)]
    return [DiffractionDataset.from_image(temp_files[k * no_images], processed_data_dir=p) for k in range(no_datasets)]


@pytest.mark.parametrize(
        'temp_files, is_h5', [
        (cbf_images, False),
        (h5_images,  True)
        ], indirect=['temp_files'], ids=['cbf', 'h5']
    )
class TestAutoPROCJob:

    def test_init(self, datasets, is_h5):
        config = AutoPROCConfig()
        job = AutoPROCJob(datasets=datasets, config=config)
        assert job._job_type == 'xtl.autoproc.process'
        assert hasattr(job, '_datasets')
        assert hasattr(job, '_config')
        assert hasattr(job, '_single_sweep')
        assert hasattr(job, '_common_config')
        assert job._is_h5 if is_h5 else not job._is_h5
        assert hasattr(job, '_run_no')
        assert hasattr(job, '_success')
        assert hasattr(job, '_results')
        assert job._success_file == 'staraniso_alldata-unique.mtz'
        assert job._shell == BashShell
        assert job._supported_shells == [BashShell]

    def test_run_no(self, datasets, is_h5):
        # Default initialization
        config = AutoPROCConfig()
        job = AutoPROCJob(datasets=datasets, config=config)
        assert job._run_no == 1

        # Existing output_dir
        d = datasets[0]
        for run in range(5):
            output_dir = (d.processed_data / f'autoproc_run{run+1:02d}')
            output_dir.mkdir(parents=True, exist_ok=True)
        assert job._determine_run_no() == 6

        # More than 99 runs
        for run in range(100):
            output_dir = (d.processed_data / f'autoproc_run{run+1:02d}')
            output_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileExistsError):
            job._determine_run_no()

    @seed(42)
    def test_patch_datasets(self, datasets, is_h5):
        config = AutoPROCConfig()
        job = AutoPROCJob(datasets=datasets, config=config)
        assert job._datasets[0].sweep_id == 1
        assert job._datasets[0].autoproc_id == 'xtl0409s01'
        if not is_h5:
            assert job._datasets[0].autoproc_idn == \
                   f'xtl0409s01,{datasets[0].raw_data},' \
                   f'{",".join(map(str, datasets[0].get_image_template(first_last=True)))}'
        assert job._datasets[0].output_dir == datasets[0].output_dir

    @seed(42)
    def test_get_macro_content(self, datasets, is_h5):
        config = AutoPROCConfig(unit_cell=[78, 78, 37, 90, 90, 90], space_group='P 43 21 2', xds_njobs=16,
                                batch_mode=True, resolution_cutoff_criterion='CC1/2', beamline='PetraIIIP14',
                                extra_params={'autoPROC_XdsIntegPostrefNumCycle': 5})
        job = AutoPROCJob(datasets=datasets, config=config)
        content = job._get_macro_content()
        print(content)
        for line in content.splitlines():
            if line.startswith('# autoproc_id'):
                assert line == f'# autoproc_id = xtl0409'
            if line.startswith('# no_sweeps'):
                assert line == f'# no_sweeps = 2'
            if line.startswith('__args'):
                assert '-B -M PetraIIIP14 -M HighResCutOnCChalf' in line
                line = line.replace('-B -M PetraIIIP14 -M HighResCutOnCChalf', '')
                prefix = '-h5' if is_h5 else '-Id'
                args = line[8:-1].split(prefix)
                ids = [arg.replace('"', '').replace(' ', '').split(',') for arg in args if arg]
                for i, id in enumerate(ids):
                    if is_h5:
                        assert Path(id[0]).parts[-4:] == ('a', 'b', 'c', f'dataset_1_{i+1}_master.h5')
                    else:
                        autoproc_id, raw_data, image_template, img_no_first, img_no_last = id
                        assert autoproc_id == f'xtl0409s0{i+1}'
                        assert Path(raw_data).parts[-3:] == ('a', 'b', 'c')
                        assert image_template == f'dataset_1_{i+1}_####.cbf'
                        assert img_no_first == '1'
                        assert img_no_last == '100'
            if line.startswith('cell'):
                assert line == f'cell="78.0 78.0 37.0 90.0 90.0 90.0"'
            if line.startswith('symm'):
                assert line == f'symm=P43212'
            if line.startswith('autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS'):
                assert line == f'autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS=16'
            if line.startswith('autoPROC_XdsIntegPostrefNumCycle'):
                assert line == f'autoPROC_XdsIntegPostrefNumCycle=5'
            if line.startswith('# job_type'):
                assert line == f'# job_type = xtl.autoproc.process'
            if line.startswith('# run_number'):
                assert line == f'# run_number = 1'
            if line.startswith('# job_dir'):
                assert Path(line.replace('# job_dir', '')).parts[-5:] == \
                       ('a', 'b', 'processed', 'c', 'autoproc_run01')
            if line.startswith('# autoproc_output_dir'):
                assert Path(line.replace('# autoproc_output_dir', '')).parts[-6:] == \
                       ('a', 'b', 'processed', 'c', 'autoproc_run01', 'autoproc')
            if line.startswith('# single_sweep'):
                assert line == f'# single_sweep = False'
            if line.startswith('# is_h5'):
                assert line == f'# is_h5 = {is_h5}'
            if line.startswith('# modules'):
                assert line == f'# modules = []'