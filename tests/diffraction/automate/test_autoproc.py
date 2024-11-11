import pytest

from xtl.diffraction.automate.autoproc import AutoPROCJobConfig2, AutoPROCJob2, AutoPROCJobResults


class TestAutoPROCJob:

    _images = [f'./a/b/c/dataset_1_1_{i+1:04d}.cbf' for i in range(0, 100)]

    @pytest.mark.make_temp_files(*_images)
    def test_init(self, temp_files):
        first_image = temp_files[0]
        config = AutoPROCJobConfig2(
            raw_data_dir=first_image.parent.parent,
            processed_data_dir=first_image.parent / 'processed',
            dataset_subdir=first_image.parent.name,
            dataset_name='dataset_1_1',
            first_image=first_image.name,
        )
        assert config.raw_data_dir == first_image.parent.parent
        assert config.processed_data_dir == first_image.parent / 'processed'
        assert config.dataset_subdir == 'c'
        assert config.dataset_name == 'dataset_1_1'
        assert config.first_image == 'dataset_1_1_0001.cbf'

        job = AutoPROCJob2(config)
        assert job._job_type == 'xtl.autoPROC.process'
