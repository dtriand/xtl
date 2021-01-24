from math import sin as _sin, sinh as _sinh, asin as _asin
from math import cos as _cos, cosh as _cosh, acos as _acos
from math import tan as _tan, tanh as _tanh, atan as _atan
from math import radians, degrees

from typing_extensions import Literal


def sin(x, mode: Literal['r', 'd'] = 'r'): return sin_d(x) if mode == 'd' else sin_r(x)


def sin_r(x): return _sin(x)


def sin_d(x): return _sin(radians(x))


def asin(x, mode: Literal['r', 'd'] = 'r'): return asin_d(x) if mode == 'd' else asin_r(x)


def asin_r(x): return _asin(x)


def asin_d(x): return degrees(_asin(x))


def cos(x, mode: Literal['r', 'd'] = 'r'): return cos_d(x) if mode == 'd' else cos_r(x)


def cos_r(x): return _cos(x)


def cos_d(x): return _cos(radians(x))


def acos(x, mode: Literal['r', 'd'] = 'r'): return acos_d(x) if mode == 'd' else acos_r(x)


def acos_r(x): return _acos(x)


def acos_d(x): return degrees(_acos(x))


def tan(x, mode: Literal['r', 'd'] = 'r'): return tan_d(x) if mode == 'd' else tan_r(x)


def tan_r(x): return _tan(x)


def tan_d(x): return _tan(radians(x))


def atan(x, mode: Literal['r', 'd'] = 'r'): return atan_d(x) if mode == 'd' else atan_r(x)


def atan_r(x): return _atan(x)


def atan_d(x): return degrees(_atan(x))
