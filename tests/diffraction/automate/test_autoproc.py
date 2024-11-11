import pytest

from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.diffraction.automate.autoproc import AutoPROCJobConfig2, AutoPROCJob2, AutoPROCJobResults


class TestAutoPROCJob:

    _images = [f'./a/b/c/dataset_1_{j}_{i+1:04d}.cbf' for j in range(1, 3)  for i in range(0, 100)]

    @pytest.mark.make_temp_files(*_images)
    def test_init(self, temp_files):
        d1 = DiffractionDataset.from_image(temp_files[0])
        d2 = DiffractionDataset.from_image(temp_files[100])
        assert d1.dataset_name != d2.dataset_name

        config = AutoPROCJobConfig2()
        job = AutoPROCJob2(datasets=[d1, d2], config=config)
        assert job._job_type == 'xtl.autoPROC.process'
