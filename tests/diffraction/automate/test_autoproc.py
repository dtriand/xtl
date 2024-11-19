import pytest

from tests.conftest import seed
from xtl.automate.shells import BashShell
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.diffraction.automate.autoproc import AutoPROCJob2, AutoPROCJobResults
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
        job = AutoPROCJob2(datasets=datasets, config=config)
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
        job = AutoPROCJob2(datasets=datasets, config=config)
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
        job = AutoPROCJob2(datasets=datasets, config=config)
        assert job._datasets[0].sweep_id == 1
        assert job._datasets[0].autoproc_id == 'xtl_0409_s01'
        if not is_h5:
            assert job._datasets[0].autoproc_idn == \
                   f'xtl_0409_s01,{datasets[0].raw_data},' \
                   f'{",".join(map(str, datasets[0].get_image_template(first_last=True)))}'
        assert job._datasets[0].job_dir == datasets[0].processed_data / 'autoproc_run01'
