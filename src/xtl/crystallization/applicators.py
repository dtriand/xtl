from dataclasses import dataclass
from enum import Enum

import numpy as np


class ReagentApplicatorType(Enum):
    CONSTANT = 'constant'
    GRADIENT = 'gradient'


class GradientApplicationMethod(Enum):
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

    def __init__(self, value: float):
        self.name = ReagentApplicatorType.CONSTANT
        self.value = value

    def __post_init__(self):
        self.min_value = self.value
        self.max_value = self.value

    def apply(self, shape: tuple[int, int]):
        # Create an array of a constant value and flatten it
        data = np.full(shape, self.value).ravel()
        return data


@dataclass
class GradientApplicator(_ReagentApplicator):
    min_value: float
    max_value: float

    def __init__(self, min_value: float, max_value: float,
                 application: str | GradientApplicationMethod = GradientApplicationMethod.HORIZONTAL,
                 scale: str | GradientScale = GradientScale.LINEAR, reverse: bool = False):
        self.name = ReagentApplicatorType.GRADIENT
        self.min_value = min_value
        self.max_value = max_value

        self.application = GradientApplicationMethod(application)
        self.scale = GradientScale(scale)
        self.reverse = reverse

    def __post_init__(self):
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

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
        if self.application == GradientApplicationMethod.HORIZONTAL:
            no_steps = cols
        elif self.application == GradientApplicationMethod.VERTICAL:
            no_steps = rows
        elif self.application == GradientApplicationMethod.CONTINUOUS:
            no_steps = rows * cols
        else:
            raise ValueError(f'Invalid gradient application: {self.application}')

        # Reverse min/max values if necessary
        min_value, max_value = self.min_value, self.max_value
        if self.reverse:
            min_value, max_value = max_value, min_value

        # Calculate datapoints
        data = spacefunc(start=min_value, stop=max_value, num=no_steps, endpoint=True)

        # Perform appropriate tiling for horizontal and vertical gradients
        if self.application == GradientApplicationMethod.HORIZONTAL:
            data = np.tile(data, (rows, 1)).ravel()
        elif self.application == GradientApplicationMethod.VERTICAL:
            data = np.tile(data, (cols, 1)).T.ravel()

        return data
