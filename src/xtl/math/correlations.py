import numpy as np
import scipy as sc


def _check_arrays_for_dcf(x1: np.array, y1: np.array, e1: np.array, x2: np.array, y2: np.array, e2: np.array):
    if x1.shape != y1.shape:
        raise ValueError('Arrays x1 and y1 must have the same shape')
    if x1.shape != e1.shape:
        raise ValueError('Arrays x1 and e1 must have the same shape')
    if x2.shape != y2.shape:
        raise ValueError('Arrays x2 and y2 must have the same shape')
    if x2.shape != e2.shape:
        raise ValueError('Arrays x2 and e2 must have the same shape')
    if len(x1.shape) != 1:
        raise ValueError('Arrays x1, y1 and e1 must be 1D arrays')
    if len(x2.shape) != 1:
        raise ValueError('Arrays x2, y2 and e2 must be 1D arrays')


def _bins_for_dcf(x_min: float, x_max: float, dx: float = None):
    if dx:
        n_steps = int((x_max - x_min) / dx)
    else:
        n_steps = 200
    return np.linspace(x_min, x_max, n_steps)


def discrete_correlation_function(x1: np.array, x2: np.array, y1: np.array, y2: np.array, e1: np.array = None,
                                  e2: np.array = None, x_min: float = None, x_max: float = None, dx: float = None):
    # Initialize error arrays as zeros if not provided
    if e1 is None:
        e1 = np.zeros_like(x1)
    if e2 is None:
        e2 = np.zeros_like(x2)
    _check_arrays_for_dcf(x1=x1, y1=y1, e1=e1, x2=x2, y2=y2, e2=e2)

    # Calculate binning step
    if x_min is None:
        x_min = np.min((x1.min(), x2.min()))
    if x_max is None:
        x_max = np.max((x1.max(), x2.max()))
    x = _bins_for_dcf(x_min=x_min, x_max=x_max, dx=dx)
    dx = x[1] - x[0]

    # Calculate lag array
    lag = np.subtract.outer(x1, x2)

    # Initialize arrays
    dcf = np.zeros_like(x)
    dcf_errors = np.zeros_like(x)
    m = np.zeros_like(x)

    # DCF calculation
    for k in range(x.shape[0]):
        x_low = x[k] - dx / 2.
        x_high = x[k] + dx / 2.
        i1, i2 = np.where((lag <= x_high) & (lag > x_low))

        y1_mean = np.mean(y1[i1])
        y2_mean = np.mean(y2[i2])
        m[k] = i1.shape[0]

        w = np.sqrt((np.var(y1[i1]) - np.mean(e1[i1]) ** 2) * (np.var(y2[i2]) - np.mean(e2[i2]) ** 2))
        dcfs = (y1[i1] - y1_mean) * (y2[i2] - y2_mean) / w
        dcf[k] = np.sum(dcfs)
        dcf_errors[k] = np.sqrt(np.sum(np.square(dcfs - dcf[k])))

    dcf /= m
    dcf_errors /= m - 1

    return x, dcf, dcf_errors


dcf1 = discrete_correlation_function


def dcf2(x1: np.array, x2: np.array, y1: np.array, y2: np.array, e1: np.array = None,
                                  e2: np.array = None, x_min: float = None, x_max: float = None, dx: float = None):
    # Initialize error arrays as zeros if not provided
    if e1 is None:
        e1 = np.zeros_like(x1)
    if e2 is None:
        e2 = np.zeros_like(x2)
    _check_arrays_for_dcf(x1=x1, y1=y1, e1=e1, x2=x2, y2=y2, e2=e2)

    # Calculate binning step
    if x_min is None:
        x_min = np.min((x1.min(), x2.min()))
    if x_max is None:
        x_max = np.max((x1.max(), x2.max()))
    x = _bins_for_dcf(x_min=x_min, x_max=x_max, dx=dx)
    dx = x[1] - x[0]

    # Calculate lag array
    lag = np.subtract.outer(x1, x2)

    # Initialize arrays
    dcf = np.zeros_like(x)
    dcf_errors = np.zeros_like(x)
    m = np.zeros_like(x)

    # DCF calculation
    for k in range(x.shape[0]):
        x_low = x[k] - dx / 2.
        x_high = x[k] + dx / 2.
        i1, i2 = np.where((lag <= x_high) & (lag > x_low))

        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)
        m[k] = i1.shape[0]

        w = np.sqrt((np.var(y1) - np.mean(e1) ** 2) * (np.var(y2) - np.mean(e2) ** 2))
        dcfs = (y1[i1] - y1_mean) * (y2[i2] - y2_mean) / w
        dcf[k] = np.sum(dcfs)
        dcf_errors[k] = np.sqrt(np.sum(np.square(dcfs - dcf[k])))

    dcf /= m
    dcf_errors /= m - 1

    return x, dcf, dcf_errors


def dcf3(x1: np.array, x2: np.array, y1: np.array, y2: np.array, e1: np.array = None,
                                  e2: np.array = None, x_min: float = None, x_max: float = None, dx: float = None):
    # Initialize error arrays as zeros if not provided
    if e1 is None:
        e1 = np.zeros_like(x1)
    if e2 is None:
        e2 = np.zeros_like(x2)
    _check_arrays_for_dcf(x1=x1, y1=y1, e1=e1, x2=x2, y2=y2, e2=e2)

    # Calculate binning step
    if x_min is None:
        x_min = np.min((x1.min(), x2.min()))
    if x_max is None:
        x_max = np.max((x1.max(), x2.max()))
    x = _bins_for_dcf(x_min=x_min, x_max=x_max, dx=dx)
    x = x[:-1] + np.diff(x) / 2
    dx = x[1] - x[0]

    # Calculate lag array
    lag = np.subtract.outer(x1, x2)

    # Initialize arrays
    dcf = np.zeros_like(x)
    dcf_errors = np.zeros_like(x)
    m = np.zeros_like(x)

    # Calculate UDCF
    y1_mean = np.mean(y1)
    y2_mean = np.mean(y2)
    w = np.sqrt((np.var(y1) - np.mean(e1) ** 2) * (np.var(y2) - np.mean(e2) ** 2))
    udcf = np.multiply.outer(y1 - y1_mean, y2 - y2_mean) / w

    # DCF calculation
    for k in range(x.shape[0]):
        x_low = x[k] - dx / 2.
        x_high = x[k] + dx / 2.
        i = np.where((lag < x_high) & (lag >= x_low))

        dcfs = udcf[i]
        m[k] = dcfs.size
        dcf[k] = np.sum(dcfs)
        dcf_errors[k] = np.sqrt(np.sum(np.square(dcfs - dcf[k])))

    dcf /= m
    dcf_errors /= m - 1

    return x, dcf, dcf_errors


def dcf4(x1: np.array, x2: np.array, y1: np.array, y2: np.array, e1: np.array = None,
                                  e2: np.array = None, x_min: float = None, x_max: float = None, dx: float = None):
    # Initialize error arrays as zeros if not provided
    if e1 is None:
        e1 = np.zeros_like(x1)
    if e2 is None:
        e2 = np.zeros_like(x2)
    _check_arrays_for_dcf(x1=x1, y1=y1, e1=e1, x2=x2, y2=y2, e2=e2)

    # Calculate binning step
    if x_min is None:
        x_min = np.min((x1.min(), x2.min()))
    if x_max is None:
        x_max = np.max((x1.max(), x2.max()))
    x = _bins_for_dcf(x_min=x_min, x_max=x_max, dx=dx)
    dx = x[1] - x[0]

    # Calculate lag array
    lag = np.subtract.outer(x1, x2)

    # Initialize arrays
    dcf = np.zeros_like(x)
    dcf_errors = np.zeros_like(x)
    m = np.zeros_like(x)

    # Calculate UDCF
    y1_mean = np.mean(y1)
    y2_mean = np.mean(y2)
    w = np.sqrt((np.var(y1) - np.mean(e1) ** 2) * (np.var(y2) - np.mean(e2) ** 2))
    udcf = np.multiply.outer(y1 - y1_mean, y2 - y2_mean) / w

    # DCF calculation
    # ibins = np.digitize(x=lag, bins=x)
    dcf, x, _ = sc.stats.binned_statistic(x=lag.flatten(), values=udcf.flatten(), bins=x, statistic='mean')
    x = x[:-1] + np.diff(x) / 2

    # for k in range(x.shape[0]):
    #     x_low = x[k] - dx / 2.
    #     x_high = x[k] + dx / 2.
    #     i = np.where((lag <= x_high) & (lag > x_low))
    #
    #     dcfs = udcf[i]
    #     m[k] = dcfs.size
    #     dcf[k] = np.sum(dcfs)
    #     dcf_errors[k] = np.sqrt(np.sum(np.square(dcfs - dcf[k])))
    #
    # dcf /= m
    # dcf_errors /= m - 1

    return x, dcf, dcf_errors
