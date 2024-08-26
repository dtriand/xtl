import numpy as np

from xtl.io.npx import NpxFile


def generate_test_data_reshape(shape: tuple[int, int], data_shape: tuple[int, int]):
    shape = np.array(shape)

    # Generate a random position for the data array
    r1 = np.random.choice(shape[0] - data_shape[0] - 1, 1)[0]
    c1 = np.random.choice(shape[1] - data_shape[1] - 1, 1)[0]
    r2 = r1 + data_shape[0] - 1
    c2 = c1 + data_shape[1] - 1

    # Generate a random data array
    data = np.random.rand(*data_shape)

    # Generate a random mask
    mask = np.ones(data_shape)
    nans_num = max(int(mask.size / 10), 1)
    nans_indices = np.random.choice(mask.size, nans_num, replace=False)
    mask.ravel()[nans_indices] = np.nan

    # Calculate location map
    location_map = np.full(shape, False)
    location_map[r1:r2+1, c1:c2+1] = True

    # Calculate the reshaped arrays
    data_reshaped = np.full(shape, np.nan)
    data_reshaped[r1:r2+1, c1:c2+1] = data * mask

    mask_reshaped = np.full(shape, np.nan)
    mask_reshaped[r1:r2+1, c1:c2+1] = mask

    # Save results to an NPX file
    npx = NpxFile(**{
                      'shape': shape,
                      'indices': np.array([[r1, c1], [r2, c2]]),
                      'data': data,
                      'location_map': location_map,
                      'mask': mask,
                      'data_reshaped': data_reshaped,
                      'mask_reshaped': mask_reshaped
                  })
    return npx


if __name__ == '__main__':
    npx1 = generate_test_data_reshape(shape=(8, 12), data_shape=(3, 3))
    npx1._header += [
        'Test case 1 for CrystallizationExperiment._fit_array',
        '- shape=(8, 12), data_shape=(5, 7)',
        '- Rectangular data array',
    ]
    npx1.save('reshape_data_ex1.npx')

    npx2 = generate_test_data_reshape(shape=(8, 12), data_shape=(4, 4))
    npx2._header += [
        'Test case 2 for CrystallizationExperiment._fit_array',
        '- shape=(8, 12), data_shape=(8, 1)',
        '- Single column data array',
    ]
    npx2.save('reshape_data_ex2.npx')

    npx3 = generate_test_data_reshape(shape=(8, 12), data_shape=(1, 8))
    npx3._header += [
        'Test case 3 for CrystallizationExperiment._fit_array',
        '- shape=(8, 12), data_shape=(1, 8)',
        '- Single row data array',
    ]
    npx3.save('reshape_data_ex3.npx')
