from typing import Iterable

import numpy as np
import pandas as pd
from pandas._typing import ArrayLike


class SpectrumData:

    def __init__(self, data: ArrayLike | Iterable | dict[str, Iterable] | pd.DataFrame):
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 2:
                raise ValueError('DataFrame must have exactly two columns.')

            if data.shape[0] == 2:
                self._data = data.values
            elif data.shape[1] == 2:
                self._data = data.values.T
            else:
                raise ValueError('DataFrame must have two columns.')
        elif isinstance(data, ArrayLike):
            if data.ndim != 2:
                raise ValueError('Data must be a 2D array.')

            if data.shape[0] == 2:
                self._data = data
            elif data.shape[1] == 2:
                self._data = data.T
            else:
                raise ValueError('Data must have two columns.')
        elif isinstance(data, dict):
            if len(data) != 2:
                raise ValueError('Dictionary must have exactly two keys.')

            x = data.get('x', [])
            y = data.get('y', [])
            if not isinstance(x, Iterable) or not isinstance(y, Iterable):
                raise ValueError('Values in dictionary must be iterable.')
            if len(x) != len(y):
                raise ValueError('Values in dictionary must have the same length.')
            if len(x) == 0 or len(y) == 0:
                raise ValueError('Values in dictionary must not be empty.')


        elif isinstance(data, Iterable):
            ...

