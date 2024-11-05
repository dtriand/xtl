from pathlib import Path

import pytest

from tests.diffraction.images.conftest import make_temp_files
from xtl.diffraction.images.datasets import DiffractionDataset


class TestDiffractionDataset:

    class TestMultifileImages:
        @pytest.mark.parametrize(
            'make_temp_files,                            dataset_name,  dataset_dir, file_ext', [
            # (first_image, raw_data_dir)
            (('./a/b/c/dataset_1_3_0001.cbf',  './a/b'), 'dataset_1_3', 'c',         '.cbf'),
            (('./a/b/dataset_1_4_0001.cbf.gz', './a'  ), 'dataset_1_4', 'b',         '.cbf.gz'),
            ], indirect=['make_temp_files'])
        def test_from_image_standard(self, make_temp_files, dataset_name, dataset_dir, file_ext):
            image, raw_data_dir = make_temp_files  # create temporary files by passing them through the fixture first
            d = DiffractionDataset.from_image(image=image)
            assert d.first_image == image
            assert d.dataset_name == dataset_name
            assert d.dataset_dir == dataset_dir
            assert d.raw_data_dir == raw_data_dir
            assert d._file_ext == file_ext

        @pytest.mark.parametrize(
            'make_temp_files,                                                        dataset_name, dataset_dir, file_ext', [
            # (first_image,                        raw_data_dir, processed_data_dir)
            (('./a/b/c/dataset_1_3_0001.cbf',      None,         './a/b/processed'), 'dataset_1_3', 'c',        '.cbf'),
            (('./a/b/c/d/dataset_1_4_0001.cbf.gz', './a/b',      None),              'dataset_1_4', 'c/d',      '.cbf.gz'),
            ],
            indirect=['make_temp_files'])
        def test_from_image_hybrid(self, make_temp_files, dataset_name, dataset_dir, file_ext):
            image, raw_data_dir, processed_data_dir = make_temp_files
            d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir,
                                              processed_data_dir=processed_data_dir)
            assert d.first_image == image
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
            ('./a/b/c/dataset_1_1_00001.cbf', './a/b/c'),
            ('./a/b/c/dataset_1_2_00001.cbf', './a/b/c/d'),
            ('./a/b/c/dataset_1_3_00001.cbf', './a/b/c/d/e'),
            ], indirect=['make_temp_files'])
        def test_from_first_image_fail(self, make_temp_files):
            image, raw_data_dir = make_temp_files
            with pytest.raises(ValueError) as e:
                d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir)
                assert "Invalid 'raw_dataset_dir' provided:" in str(e.value)

        @pytest.mark.parametrize(
            'make_temp_files,                                                               dataset_name,  dataset_dir, file_ext', [
            # (first_image, other_image, raw_data_dir)
            (('./a/b/c/dataset_1_3_0001.cbf',  './a/b/c/dataset_1_3_0002.cbf',    './a/b'), 'dataset_1_3', 'c',         '.cbf'),
            (('./a/b/dataset_1_4_0002.cbf.gz', './a/b/dataset_1_4_0003.cbf.gz',   './a'),   'dataset_1_4', 'b',         '.cbf.gz'),
            ], indirect=['make_temp_files']
        )
        def test_from_image_other_than_first(self, make_temp_files, dataset_name, dataset_dir, file_ext):
            first_image, other_image, raw_data_dir = make_temp_files
            d = DiffractionDataset.from_image(image=other_image)
            assert d.first_image == first_image
            assert d.dataset_name == dataset_name
            assert d.dataset_dir == dataset_dir
            assert d.raw_data_dir == raw_data_dir
            assert d._file_ext == file_ext

        @pytest.mark.parametrize(
            'make_temp_files, dataset_name', [
            (('./a/b/c/dataset_1_1_0001.cbf',), 'dataset_1_1'),
            (('./a/b/dataset_1.2_0002',), 'dataset_1.2'),
            (('./a/b/dataset_1.3_0003.cbf',), 'dataset_1.3'),
            (('./a/b/dataset.1_4_0004.cbf',), 'dataset.1_4'),
            (('./a/b/dataset.1_5_0005.cbf.gz',), 'dataset.1_5'),
            ], indirect=['make_temp_files']
        )
        def test_file_instead_of_dataset_name(self, make_temp_files, dataset_name):
            image = make_temp_files[0]
            d = DiffractionDataset(dataset_name=image.name, dataset_dir=image.parent.name,
                                   raw_data_dir=image.parent.parent)
            assert d.dataset_name == dataset_name


    class TestH5Images:

        @pytest.mark.parametrize(
            'make_temp_files,                                                                                                                       dataset_name, dataset_dir, file_ext',
            [
                # (*images, raw_data_dir, processed_data_dir)
                (('./a/b/c/dataset_1_1_master.h5', './a/b/c/dataset_1_1_data_001.h5', './a/b/c/dataset_1_1_data_001.h5', './a', None),              'dataset_1_1', 'b/c',      '.h5'),
                (('./a/b/c/dataset_1_2_master.h5', './a/b/c/dataset_1_2_data_001.h5', './a/b/c/dataset_1_2_data_001.h5', None,  './a/b/processed'), 'dataset_1_2', 'c',        '.h5'),
            ],
            indirect=['make_temp_files'])
        def test_from_image_master_or_data(self, make_temp_files, dataset_name, dataset_dir, file_ext):
            files = make_temp_files
            raw_data_dir, processed_data_dir = files[-2:]
            images = files[:-2]
            for image in images:
                # Try initializing the dataset from each of the images and check if the master file is discovered
                d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir,
                                                  processed_data_dir=processed_data_dir)
                assert d.first_image == images[0]
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
        def test_from_image_invalid_raw_dir(self, make_temp_files):
            image, raw_data_dir = make_temp_files
            with pytest.raises(ValueError) as e:
                d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir)
                assert "Invalid 'raw_dataset_dir' provided:" in str(e.value)

        @pytest.mark.parametrize(
            'make_temp_files, dataset_name, fmt', [
            (('./a/b/c/dataset_1_1_master.h5', './a/b/dataset_1_1_data_0001.h5',), 'dataset_1_1', '.h5'),
            (('./a/b/dataset_1_2_data_0002.h5', './a/b/dataset_1_2_master.h5',),   'dataset_1_2', '.h5'),
            (('./a/b/dataset_1.3_data_0003.h5', './a/b/dataset_1.3_master.h5',),   'dataset_1.3', '.h5'),
            ], indirect=['make_temp_files']
        )
        def test_file_instead_of_dataset_name(self, make_temp_files, dataset_name, fmt):
            images = make_temp_files
            d = DiffractionDataset(dataset_name=images[0].name, dataset_dir=images[0].parent.name,
                                   raw_data_dir=images[0].parent.parent, fmt=fmt)
            assert d.dataset_name == dataset_name


    class TestFStringFactory:

        @pytest.mark.parametrize(
            'make_temp_files, fstring, keys', [
            (('./a/b/c/dataset_1_1_0001.cbf', ), '{raw_data_dir}/{dataset_dir}/{dataset_name}', ['raw_data_dir', 'dataset_dir', 'dataset_name']),
            (('./a/b/c/dataset_1_1_0001.cbf', ), '{dataset_dir}/{dataset_name}', ['dataset_dir', 'dataset_name']),
            (('./a/b/c/dataset_1_1_0001.cbf', ), '{dataset_name}', ['dataset_name']),
            (('./a/b/c/dataset_1_1_0001.cbf', ), '{dataset_name}/a_custom_dir', ['dataset_name']),
        ], indirect=['make_temp_files'])
        def test_register_fstring(self, make_temp_files, fstring, keys):
            image = make_temp_files[0]
            d = DiffractionDataset.from_image(image=image)

            # Normal use
            d.register_dir_fstring(dir_type='a_custom_dir', fstring=fstring, keys=keys)
            assert hasattr(d, 'a_custom_dir')

            # Additional key in the validator, i.e. missing key in the f-string
            extra_keys = keys + ['extra_key']
            with pytest.raises(ValueError, match='Missing key') as e:
                d.register_dir_fstring(dir_type='a_custom_dir_with_extra_key', fstring=fstring, keys=extra_keys)

            # Missing key in the validator, i.e. unexpected key in the f-string
            missing_keys = keys[:-1]
            with pytest.raises(ValueError, match='Unexpected key') as e:
                d.register_dir_fstring(dir_type='a_custom_dir_with_missing_key', fstring=fstring, keys=missing_keys)