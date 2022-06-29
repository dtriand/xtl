import numpy as np
from scipy import signal as ss


def moving_average(x: np.ndarray, w: int = 3):
    # np.convolve returns sum(x[i:i+w] * 1) for i in range(len(x)-(w-1))
    return np.convolve(x, np.ones(w), mode='same') / w


def savitzky_golay(x: np.ndarray, w: int = 5, n: int = 3, **kwargs):
    return ss.savgol_filter(x, window_length=w, polyorder=n, **kwargs)
