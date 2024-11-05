from pathlib import Path

import pytest

from xtl.diffraction.images.datasets import DiffractionDataset


class TestDiffractionDataset:

    @pytest.mark.parametrize(
        'make_temp_files,                            dataset_name,  dataset_dir, file_ext', [
        # (first_image, raw_data_dir)
        (('./a/b/c/dataset_1_1_master.h5', './a/b'), 'dataset_1_1', 'c',         '.h5'),
        (('./a/b/dataset_1_2_data_001.h5', './a'  ), 'dataset_1_2', 'b',         '.h5'),
        (('./a/b/c/dataset_1_3_0001.cbf',  './a/b'), 'dataset_1_3', 'c',         '.cbf'),
        (('./a/b/dataset_1_4_0001.cbf.gz', './a'  ), 'dataset_1_4', 'b',         '.cbf.gz'),
        ], indirect=['make_temp_files'])
    def test_from_first_image_standard(self, make_temp_files, dataset_name, dataset_dir, file_ext):
        image, raw_data_dir = make_temp_files  # create temporary files by passing them through the fixture first
        if file_ext == '.h5':
            with pytest.raises(NotImplementedError):
                d = DiffractionDataset.from_image(image=image)
            return
        d = DiffractionDataset.from_image(image=image)
        assert d.first_image == image.name
        assert d.dataset_name == dataset_name
        assert d.dataset_dir == dataset_dir
        assert d.raw_data_dir == raw_data_dir
        assert d._file_ext == file_ext

    @pytest.mark.parametrize(
        'make_temp_files,                                                        dataset_name, dataset_dir, file_ext', [
        # (first_image,                        raw_data_dir, processed_data_dir)
        (('./a/b/c/dataset_1_1_master.h5',     './a/b',      './a/b/processed'), 'dataset_1_1', 'c',        '.h5'),
        (('./a/b/dataset_1_2_data_001.h5',     './a',        './a/processed'),   'dataset_1_2', 'b',        '.h5'),
        (('./a/b/c/dataset_1_3_0001.cbf',      None,         None),              'dataset_1_3', 'c',        '.cbf'),
        (('./a/b/c/d/dataset_1_4_0001.cbf.gz', './a/b',      './a/b/processed'), 'dataset_1_4', 'c/d',      '.cbf.gz'),
        ],
        indirect=['make_temp_files'])
    def test_from_first_image_hybrid(self, make_temp_files, dataset_name, dataset_dir, file_ext):
        image, raw_data_dir, processed_data_dir = make_temp_files
        if file_ext == '.h5':
            with pytest.raises(NotImplementedError):
                d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir,
                                                  processed_data_dir=processed_data_dir)
            return
        d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir,
                                          processed_data_dir=processed_data_dir)
        assert d.first_image == image.name
        assert d.dataset_name == dataset_name
        assert d.dataset_dir == dataset_dir
        if raw_data_dir is None:
            assert d.raw_data_dir == image.parent.parent
        else:
            assert d.raw_data_dir == raw_data_dir
        if processed_data_dir is None:
            assert d.processed_data_dir == Path('.')
        else:
            assert d.processed_data_dir == processed_data_dir
        assert d._file_ext == file_ext

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize(
        'make_temp_files', [
        ('./a/b/c/dataset_1_1_master.h5', './a/b/c'),
        ('./a/b/c/dataset_1_2_master.h5', './a/b/c/d'),
        ('./a/b/c/dataset_1_3_master.h5', './a/b/c/d/e'),
        ], indirect=['make_temp_files'])
    def test_from_first_image_fail(self, make_temp_files):
        image, raw_data_dir = make_temp_files
        with pytest.raises(ValueError) as e:
            d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir)
            assert "Invalid 'raw_dataset_dir' provided:" in str(e.value)

    @pytest.mark.parametrize(
        'make_temp_files,                                                               dataset_name,  dataset_dir, file_ext', [
        # (first_image, other_image, raw_data_dir)
        (('./a/b/c/dataset_1_1_master.h5', './a/b/c/dataset_1_1_data_001.h5', './a/b'), 'dataset_1_1', 'c',         '.h5'),
        (('./a/b/dataset_1_2_data_001.h5', './a/b/dataset_1_2_data_002.h5',   './a'),   'dataset_1_2', 'b',         '.h5'),
        (('./a/b/c/dataset_1_3_0001.cbf',  './a/b/c/dataset_1_3_0002.cbf',    './a/b'), 'dataset_1_3', 'c',         '.cbf'),
        (('./a/b/dataset_1_4_0002.cbf.gz', './a/b/dataset_1_4_0003.cbf.gz',   './a'),   'dataset_1_4', 'b',         '.cbf.gz'),
        ], indirect=['make_temp_files']
    )
    def test_from_image_other_than_first(self, make_temp_files, dataset_name, dataset_dir, file_ext):
        first_image, other_image, raw_data_dir = make_temp_files
        if file_ext == '.h5':
            with pytest.raises(NotImplementedError):
                d = DiffractionDataset.from_image(image=other_image, raw_dataset_dir=raw_data_dir)
            return
        d = DiffractionDataset.from_image(image=other_image)
        assert d.first_image == first_image.name
        assert d.dataset_name == dataset_name
        assert d.dataset_dir == dataset_dir
        assert d.raw_data_dir == raw_data_dir
        assert d._file_ext == file_ext