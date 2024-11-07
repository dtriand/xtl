from pathlib import Path

import pytest

from xtl.diffraction.images.datasets import DiffractionDataset


class TestDiffractionDataset:

    class TestMultifileImages:
        @pytest.mark.parametrize(
            'temp_files,                                 dataset_name,  dataset_dir, file_ext', [
            # (first_image, raw_data_dir)
            (('./a/b/c/dataset_1_3_0001.cbf',  './a/b'), 'dataset_1_3', 'c',         '.cbf'),
            (('./a/b/dataset_1_4_0001.cbf.gz', './a'  ), 'dataset_1_4', 'b',         '.cbf.gz'),
            ], indirect=['temp_files'])
        def test_from_image_standard(self, temp_files, dataset_name, dataset_dir, file_ext):
            image, raw_data_dir = temp_files  # create temporary files by passing them through the fixture first
            d = DiffractionDataset.from_image(image=image)
            assert d.first_image == image
            assert d.dataset_name == dataset_name
            assert d.dataset_dir == dataset_dir
            assert d.raw_data_dir == raw_data_dir
            assert d._file_ext == file_ext

        @pytest.mark.parametrize(
            'temp_files,                                                              dataset_name, dataset_dir, file_ext', [
            # (first_image,                        raw_data_dir, processed_data_dir)
            (('./a/b/c/dataset_1_3_0001.cbf',      None,         './a/b/processed'), 'dataset_1_3', 'c',        '.cbf'),
            (('./a/b/c/d/dataset_1_4_0001.cbf.gz', './a/b',      None),              'dataset_1_4', 'c/d',      '.cbf.gz'),
            ],
            indirect=['temp_files'])
        def test_from_image_hybrid(self, temp_files, dataset_name, dataset_dir, file_ext):
            image, raw_data_dir, processed_data_dir = temp_files
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
            'temp_files', [
            ('./a/b/c/dataset_1_1_00001.cbf', './a/b/c'),
            ('./a/b/c/dataset_1_2_00001.cbf', './a/b/c/d'),
            ('./a/b/c/dataset_1_3_00001.cbf', './a/b/c/d/e'),
            ], indirect=['temp_files'])
        def test_from_first_image_fail(self, temp_files):
            image, raw_data_dir = temp_files
            with pytest.raises(ValueError) as e:
                d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir)
                assert "Invalid 'raw_dataset_dir' provided:" in str(e.value)

        @pytest.mark.parametrize(
            'temp_files,                                                                    dataset_name,  dataset_dir, file_ext', [
            # (first_image, other_image, raw_data_dir)
            (('./a/b/c/dataset_1_3_0001.cbf',  './a/b/c/dataset_1_3_0002.cbf',    './a/b'), 'dataset_1_3', 'c',         '.cbf'),
            (('./a/b/dataset_1_4_0002.cbf.gz', './a/b/dataset_1_4_0003.cbf.gz',   './a'),   'dataset_1_4', 'b',         '.cbf.gz'),
            ], indirect=['temp_files']
        )
        def test_from_image_other_than_first(self, temp_files, dataset_name, dataset_dir, file_ext):
            first_image, other_image, raw_data_dir = temp_files
            d = DiffractionDataset.from_image(image=other_image)
            assert d.first_image == first_image
            assert d.dataset_name == dataset_name
            assert d.dataset_dir == dataset_dir
            assert d.raw_data_dir == raw_data_dir
            assert d._file_ext == file_ext

        @pytest.mark.parametrize(
            'temp_files,                      dataset_name', [
            ('./a/b/c/dataset_1_1_0001.cbf',  'dataset_1_1'),
            ('./a/b/dataset_1.2_0002',        'dataset_1.2'),
            ('./a/b/dataset_1.3_0003.cbf',    'dataset_1.3'),
            ('./a/b/dataset.1_4_0004.cbf',    'dataset.1_4'),
            ('./a/b/dataset.1_5_0005.cbf.gz', 'dataset.1_5'),
            ], indirect=['temp_files']
        )
        def test_file_instead_of_dataset_name(self, temp_files, dataset_name):
            image = temp_files
            d = DiffractionDataset(dataset_name=image.name, dataset_dir=image.parent.name,
                                   raw_data_dir=image.parent.parent)
            assert d.dataset_name == dataset_name


    class TestH5Images:

        @pytest.mark.parametrize(
            'temp_files,                                                                                                                        dataset_name,  dataset_dir, file_ext', [
            # (*images, raw_data_dir, processed_data_dir)
            (('./a/b/c/dataset_1_1_master.h5', './a/b/c/dataset_1_1_data_001.h5', './a/b/c/dataset_1_1_data_002.h5', './a', None),              'dataset_1_1', 'b/c',       '.h5'),
            (('./a/b/c/dataset_1_2_master.h5', './a/b/c/dataset_1_2_data_001.h5', './a/b/c/dataset_1_2_data_002.h5', None,  './a/b/processed'), 'dataset_1_2', 'c',         '.h5'),
            ],
            indirect=['temp_files'])
        def test_from_image_master_or_data(self, temp_files, dataset_name, dataset_dir, file_ext):
            files = temp_files
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
            'temp_files', [
            ('./a/b/c/dataset_1_1_master.h5', './a/b/c'),
            ('./a/b/c/dataset_1_2_master.h5', './a/b/c/d'),
            ('./a/b/c/dataset_1_3_master.h5', './a/b/c/d/e'),
            ], indirect=['temp_files'])
        def test_from_image_invalid_raw_dir(self, temp_files):
            image, raw_data_dir = temp_files
            with pytest.raises(ValueError) as e:
                d = DiffractionDataset.from_image(image=image, raw_dataset_dir=raw_data_dir)
                assert "Invalid 'raw_dataset_dir' provided:" in str(e.value)

        @pytest.mark.parametrize(
            'temp_files,                                                            dataset_name,  fmt', [
            (('./a/b/c/dataset_1_1_master.h5',  './a/b/dataset_1_1_data_0001.h5',), 'dataset_1_1', '.h5'),
            (('./a/b/dataset_1_2_data_0002.h5', './a/b/dataset_1_2_master.h5',),    'dataset_1_2', '.h5'),
            (('./a/b/dataset_1.3_data_0003.h5', './a/b/dataset_1.3_master.h5',),    'dataset_1.3', '.h5'),
            ], indirect=['temp_files']
        )
        def test_file_instead_of_dataset_name(self, temp_files, dataset_name, fmt):
            images = temp_files
            d = DiffractionDataset(dataset_name=images[0].name, dataset_dir=images[0].parent.name,
                                   raw_data_dir=images[0].parent.parent, fmt=fmt)
            assert d.dataset_name == dataset_name

    class TestInternals:

        @pytest.mark.parametrize(
            'filename,                  file_stem,            file_ext', [
            ('dataset_1_1_0001.cbf',    'dataset_1_1_0001',   '.cbf'),
            ('dataset_1_2_0002.cbf.gz', 'dataset_1_2_0002',   '.cbf.gz'),
            ('dataset_1_3_master.h5',   'dataset_1_3_master', '.h5'),
            ('dataset_1_4_0004',        'dataset_1_4_0004',   ''),
            ('dataset_1_5_0005.gz',     'dataset_1_5_0005',   '.gz'),
        ])
        def test_get_file_stem_and_extension(self, filename, file_stem, file_ext):
            assert DiffractionDataset._get_file_stem_and_extension(image=filename) == (file_stem, file_ext)

        @pytest.mark.parametrize(
            'filename,                   is_h5, dataset_name', [
            ('dataset_1_1_0001.cbf',     False, 'dataset_1_1'),
            ('dataset_1_2_0002.cbf.gz',  False, 'dataset_1_2'),
            ('dataset_1_3_master.h5',    True,  'dataset_1_3'),
            ('dataset_1.4_data_0004.h5', True,  'dataset_1.4'),
            ('dataset_1_5_0004',         False, 'dataset_1_5'),
            ('dataset_1.6_0005.cbf',     False, 'dataset_1.6'),
        ])
        def test_determine_dataset_name(self, filename, is_h5, dataset_name):
            assert DiffractionDataset._determine_dataset_name(filename=filename, is_h5=is_h5) == dataset_name

        @pytest.mark.make_temp_files(*[f'./a/b/c/dataset_1_1_{i+1:04d}.cbf.gz' for i in range(0, 20)],
                                     *[f'./a/b/c/dataset_1_1_{i+1:04d}' for i in range(0, 20)])
        @pytest.mark.parametrize(
            'pattern,                   files_only, expected_indices', [
            ('dataset_1_1_*.cbf.gz',    True,       (0, 20)),
            ('dataset_1_1_001*.cbf.gz', True,       (9, 19)),
            ('dataset_1_1*',            False,      (0, 40)),
            ('dataset_1_1*',            True,       (0, 20)),
        ])
        def test_glob_directory(self, temp_files, pattern, files_only, expected_indices):
            start, end = expected_indices
            assert DiffractionDataset._glob_directory(directory=temp_files[0].parent, pattern=pattern,
                                                      files_only=files_only) == sorted(temp_files[start:end])

        _multiple_cbf_datasets = [f'./a/b/c/dataset_1_{j}_{i+1:04d}.cbf' for j in range(1, 3) for i in range(0, 10)]
        _multiple_h5_datasets = sorted([f'./a/b/c/dataset_1_{j}_master.h5' for j in range(1, 3)] +
                                       [f'./a/b/c/dataset_1_{j}_data_{i+1:04d}.h5' for i in range(0, 10) for j in range(1, 3)])

        @pytest.mark.make_temp_files(*_multiple_cbf_datasets[:10])
        def test_get_all_images(self, temp_files):
            d = DiffractionDataset.from_image(image=temp_files[0])
            assert d._get_all_images() == sorted(temp_files)

        @pytest.mark.make_temp_files(*_multiple_cbf_datasets)
        def test_get_dataset_images(self, temp_files):
            # Isolate dataset images when multiple datasets are present in the directory
            d = DiffractionDataset.from_image(image=temp_files[0])
            assert d._get_dataset_images() == sorted(temp_files[0:10])

            # Check that the results are cached
            image = temp_files[0].with_name('dataset_1_1_0011.cbf')
            image.touch()
            assert d._get_dataset_images() == sorted(temp_files[0:10])

            # Reset the cache and re-glob
            d.reset_images_cache()
            assert d._get_dataset_images() == sorted(temp_files[0:10] + [image])

        @pytest.mark.make_temp_files(*_multiple_cbf_datasets)
        def test_get_images(self, temp_files):
            # Isolate dataset images when multiple datasets are present in the directory
            d = DiffractionDataset.from_image(image=temp_files[0])
            assert d.get_images() == sorted(temp_files[0:10])

            # Check that the results are not being cached
            image = temp_files[0].with_name('dataset_1_1_0011.cbf')
            image.touch()
            assert d.get_images() == sorted(temp_files[0:10] + [image])

        @pytest.mark.make_temp_files(*_multiple_h5_datasets)
        def test_get_h5_images(self, temp_files):
            # Check detection of master images
            d = DiffractionDataset.from_image(image=temp_files[0])  # dataset_1_1_data_0001.h5
            assert d._get_h5_master_images() == [temp_files[10], temp_files[21]]

            # Check detection of dataset images
            assert d._get_h5_images() == [temp_files[10], *temp_files[0:10]]

        @pytest.mark.parametrize(
            'temp_files,             is_h5', [
            (_multiple_cbf_datasets, False),
            (_multiple_h5_datasets,  True),
        ], indirect=['temp_files'])
        def test_determine_images(self, temp_files, is_h5):
            d = DiffractionDataset.from_image(image=temp_files[0])
            if is_h5:
                images = d._determine_images()
                assert sorted(images) == temp_files[0:11]
                assert 'master' in images[0].name  # first image should be the master file
                assert d.first_image.name == 'dataset_1_1_master.h5'
                assert d.last_image is None
                assert d.no_images is None
                assert d.file_extension == '.h5'
            else:
                assert d._determine_images() == temp_files[0:10]
                assert d.first_image.name == 'dataset_1_1_0001.cbf'
                assert d.last_image.name == 'dataset_1_1_0010.cbf'
                assert d.no_images == 10
                assert d.file_extension == '.cbf'

        @pytest.mark.parametrize(
            'temp_files,             is_h5', [
            (_multiple_cbf_datasets, False),
            (_multiple_h5_datasets,  True),
        ], indirect=['temp_files'])
        def test_get_all_dataset_names(self, temp_files, is_h5):
            d = DiffractionDataset.from_image(image=temp_files[0])
            assert d.get_all_dataset_names() == ['dataset_1_1', 'dataset_1_2']

    class TestFStringFactory:

        _images = ['./a/b/c/dataset_1_1_0001.cbf']

        @pytest.mark.make_temp_files(*_images)
        @pytest.mark.parametrize(
            'fstring, keys', [
            ('{raw_data_dir}/{dataset_dir}/{dataset_name}', ['raw_data_dir', 'dataset_dir', 'dataset_name']),
            ('{dataset_dir}/{dataset_name}',                ['dataset_dir', 'dataset_name']),
            ('{dataset_name}',                              ['dataset_name']),
            ('{dataset_name}/a_custom_dir',                 ['dataset_name']),
        ])
        def test_register_fstring(self, temp_files, fstring, keys):
            d = DiffractionDataset.from_image(image=temp_files)
            d.register_dir_fstring(dir_type='a_custom_dir', fstring=fstring, keys=keys)
            assert hasattr(d, 'a_custom_dir')

        @pytest.mark.make_temp_files(*_images)
        def test_register_fstring_extra_key(self, temp_files):
            d = DiffractionDataset.from_image(image=temp_files)
            with pytest.raises(ValueError, match='Missing key') as e:
                d.register_dir_fstring(dir_type='a_custom_dir', fstring='{dataset_name}', keys=['extra_key'])

        @pytest.mark.make_temp_files(*_images)
        def test_register_fstring_missing_key(self, temp_files):
            d = DiffractionDataset.from_image(image=temp_files)
            with pytest.raises(ValueError, match='Unexpected key') as e:
                d.register_dir_fstring(dir_type='a_custom_dir', fstring='{dataset_name}', keys=[])