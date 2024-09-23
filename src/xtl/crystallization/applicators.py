from dataclasses import dataclass
from enum import Enum

import numpy as np


class ReagentApplicatorType(Enum):
    CONSTANT = 'constant'
    GRADIENT = 'gradient'
    STEP_FIXED = 'step_fixed'
    STEP_LIST = 'step_list'


class ApplicationMethod(Enum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    CONTINUOUS = 'continuous'


class GradientScale(Enum):
    LINEAR = 'linear'
    LOGARITHMIC = 'logarithmic'


@dataclass
class _ReagentApplicator:
    name: ReagentApplicatorType

    def apply(self, shape: tuple[int, int]):
        raise NotImplementedError


@dataclass
class ConstantApplicator(_ReagentApplicator):
    value: float
    min_value: float
    max_value: float

    def __init__(self, value: float):
        self.name = ReagentApplicatorType.CONSTANT
        if not isinstance(value, (int, float)):
            raise TypeError(f'Value must be an integer or float, not {type(value)}')
        if value <= 0:
            raise ValueError('Value must be positive')
        self.value = float(value)
        self.min_value = self.value
        self.max_value = self.value

    def apply(self, shape: tuple[int, int]):
        # Create an array of a constant value and flatten it
        data = np.full(shape, self.value).ravel()
        return data

    def to_dict(self):
        return {
            'name': self.name.value,
            'value': self.value
        }


@dataclass
class GradientApplicator(_ReagentApplicator):
    min_value: float
    max_value: float

    def __init__(self, min_value: float, max_value: float,
                 method: str | ApplicationMethod = ApplicationMethod.HORIZONTAL,
                 scale: str | GradientScale = GradientScale.LINEAR, reverse: bool = False):
        self.name = ReagentApplicatorType.GRADIENT
        self.min_value = min_value
        self.max_value = max_value
        for value, name in zip((min_value, max_value), ('min_value', 'max_value')):
            if not isinstance(value, (int, float)):
                raise TypeError(f'Value \'{name}\' must be an integer or float, not {type(value)}')
            if value < 0:
                raise ValueError(f'Value \'{name}\'  must be positive')
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

        self.method = ApplicationMethod(method)
        self.scale = GradientScale(scale)
        self.reverse = reverse

    def apply(self, shape: tuple[int, int]):
        rows, cols = shape

        # Function for calculating the gradient
        if self.scale == GradientScale.LINEAR:
            spacefunc = np.linspace
        elif self.scale == GradientScale.LOGARITHMIC:
            spacefunc = np.geomspace
        else:
            raise ValueError(f'Invalid gradient scale: {self.scale}')

        # Calculate the number of steps
        if self.method == ApplicationMethod.HORIZONTAL:
            no_steps = cols
        elif self.method == ApplicationMethod.VERTICAL:
            no_steps = rows
        elif self.method == ApplicationMethod.CONTINUOUS:
            no_steps = rows * cols
        else:
            raise ValueError(f'Invalid ApplicationMethod: {self.method}')

        # Reverse min/max values if necessary
        min_value, max_value = self.min_value, self.max_value
        if self.reverse:
            min_value, max_value = max_value, min_value

        # Calculate datapoints
        data = spacefunc(start=min_value, stop=max_value, num=no_steps, endpoint=True)

        # Perform appropriate tiling for horizontal and vertical gradients
        if self.method == ApplicationMethod.HORIZONTAL:
            data = np.tile(data, (rows, 1)).ravel()
        elif self.method == ApplicationMethod.VERTICAL:
            data = np.tile(data, (cols, 1)).T.ravel()

        return data

    def to_dict(self):
        return {
            'name': self.name.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'method': self.method.value,
            'scale': self.scale.value,
            'reverse': self.reverse
        }

@dataclass
class StepFixedApplicator(_ReagentApplicator):
    min_value: float
    max_value: float
    step: float

    def __init__(self, start_value: float, step: float,
                 method: str | ApplicationMethod = ApplicationMethod.HORIZONTAL,
                 reverse: bool = False):
        self.name = ReagentApplicatorType.STEP_FIXED
        self.reverse = reverse

        # Type and value checking
        for value, name in zip((start_value, step), ('start_value', 'step')):
            if not isinstance(value, (int, float)):
                raise TypeError(f'Value \'{name}\' must be an integer or float, not {type(value)}')
        if start_value < 0:
            raise ValueError(f'Value \'{name}\'  must non-negative')
        if step == 0:
            raise ValueError('Step must be non-zero')

        # Assume decreasing gradient if step is negative
        self.step = float(step)
        if self.step < 0:
            self.step = -step  # step is always positive
            self.reverse = True

        # Calculate min and max values
        if not self.reverse:
            self.min_value = start_value
            self.max_value = None
        else:
            self.min_value = None
            self.max_value = start_value

        self.method = ApplicationMethod(method)

    def apply(self, shape: tuple[int, int]):
        rows, cols = shape

        # Calculate the number of steps
        if self.method == ApplicationMethod.HORIZONTAL:
            no_steps = cols
        elif self.method == ApplicationMethod.VERTICAL:
            no_steps = rows
        elif self.method == ApplicationMethod.CONTINUOUS:
            no_steps = rows * cols
        else:
            raise ValueError(f'Invalid ApplicationMethod: {self.method}')

        # Calculate missing min/max values
        if not self.reverse:
            self.max_value = self.min_value + no_steps * self.step
        else:
            self.min_value = self.max_value - (no_steps - 1) * self.step

        # Check if concentration drops below 0
        if self.min_value < 0:
            suggested_step = self.max_value / no_steps
            raise ValueError(f'Invalid \'shape\' and/or \'step_size\': Minimum concentration is less than 0.'
                             f'For the given shape, \'step_size\' should be less than {suggested_step}')

        # Calculate datapoints
        if not self.reverse:
            data = np.arange(start=self.min_value, stop=self.max_value, step=self.step)
        else:
            # stop is exclusive, so add step to max_value
            data = np.arange(start=self.min_value, stop=self.max_value + self.step, step=self.step)
            # reverse the array, so that it goes from max to min, as requested
            data = np.flip(data)

        # Perform appropriate tiling for horizontal and vertical gradients
        if self.method == ApplicationMethod.HORIZONTAL:
            data = np.tile(data, (rows, 1)).ravel()
        elif self.method == ApplicationMethod.VERTICAL:
            data = np.tile(data, (cols, 1)).T.ravel()

        return data

    def to_dict(self):
        return {
            'name': self.name.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step': self.step,
            'method': self.method.value,
            'reverse': self.reverse
        }

@dataclass
class StepListApplicator(_ReagentApplicator):
    min_value: float
    max_value: float
    steps: list[float]
