from __future__ import annotations

import colorsys
from typing import Any

from pydantic import field_validator, ValidationInfo

from xtl.common.options import Option
from .base import BaseGraphModel


class Color(BaseGraphModel):

    r: float = \
        Option(
            desc='The red value of the color (0-1)',
            ge=0., le=1.,
        )

    g: float = \
        Option(
            desc='The green value of the color (0-1)',
            ge=0., le=1.,
        )

    b: float = \
        Option(
            desc='The blue value of the color (0-1)',
            ge=0., le=1.,
        )

    alpha: float = \
        Option(
            desc='The alpha value of the color (0-1)',
            ge=0., le=1., default=1.
        )

    @field_validator('r', 'g', 'b', 'alpha', mode='before')
    @classmethod
    def _normalize(cls, value: Any, info: ValidationInfo):
        if value < 0:
            raise ValueError(f'Channel {info.field_name!r} must be >= 0, got {value!r}')
        if value > 255:
            raise ValueError(f'Channel {info.field_name!r} must be <= 255, got {value!r}')
        if value > 1.0:
            return round(value / 255.0, 10)
        return float(value)


    @staticmethod
    def _parse_hex(value: str) -> tuple[float, float, float, float]:
        """
        Parse #RGB, #RRGGBB and #RRGGBBAA strings to (r, g, b, alpha) in 0-1 range.

        :param value:
        :return:
        """
        s = value.strip().lstrip('#')
        if len(s) == 3:
            # Convert RGB -> RRGGBB
            s = ''.join([c * 2 for c in s])
        if len(s) == 6:
            # Convert RRGGBB -> RRGGBBAA
            s += 'ff'
        if len(s) != 8:
            raise ValueError(f'Invalid hex color {value!r}. Expected #RGB, #RRGGBB or #RRGGBBAA format.')

        try:
            r, g, b, a = (int(s[i:i+2], 16) / 255.0 for i in range(0, 8, 2))
        except ValueError:
            raise ValueError(f'Invalid hex color {value!r}: non-hex characters in sequence')
        return r, g, b, a

    @classmethod
    def from_hex(cls, value: str) -> Color:
        """
        Construct a Color from a hex string (#RGB, #RRGGBB or #RRGGBBAA).

        :param value:
        :return:
        """
        r, g, b, a = cls._parse_hex(value)
        return cls(r=r, g=g, b=b, alpha=a)

    @classmethod
    def from_name(cls, name: str) -> Color:
        """
        Construct a Color from a named color (e.g., 'red', 'blue', 'green').

        :param name:
        :return:
        """
        import matplotlib.colors as mcolors
        if name not in mcolors.CSS4_COLORS:
            close = [k for k in mcolors.CSS4_COLORS if k.startswith(name[:3])]
            hint = f'  Did you mean one of: {",".join(close[:5])}' if close else ''
            raise ValueError(f'Unknown CSS color name {name!r}.{hint}\n'
                             f'See: https://www.w3.org/TR/css-color-4/#named-colors')
        hex_value = mcolors.CSS4_COLORS[name]
        return cls.from_hex(hex_value)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0) -> Color:
        if not (0. <= h <= 360.):
            raise ValueError(f'Hue must be between 0 and 360, got {h!r}')
        if not (0. <= s <= 1.):
            raise ValueError(f'Saturation must be between 0 and 1, got {s!r}')
        if not (0. <= l <= 1.):
            raise ValueError(f'Lightness must be between 0 and 1, got {l!r}')
        r, g, b = colorsys.hls_to_rgb(h / 360., l, s)
        return cls(r=round(r, 10), g=round(g, 10), b=round(b, 10))

    @classmethod
    def from_cmyk(cls, c: float, m: float, y: float, k: float, a: float = 1.0) -> Color:
        for value, channel in ((c, 'c'), (m, 'm'), (y, 'y'), (k, 'k')):
            if not 0 <= value <= 1:
                raise ValueError(f'CMYK channel {channel!r} must be between 0 and 1, got {value!r}')

        r = round((1. - c) * (1. - k), 10)
        g = round((1. - m) * (1. - k), 10)
        b = round((1. - c) * (1. - k), 10)
        return cls(r=r, g=g, b=b, alpha=a)

    def to_rgb(self, as_int: bool = False) -> tuple[float, float, float]:
        if as_int:
            return int(self.r * 255), int(self.g * 255), int(self.b * 255)
        return self.r, self.g, self.b

    def to_rgba(self, as_int: bool = False) -> tuple[float, float, float, float]:
        if as_int:
            return int(self.r * 255), int(self.g * 255), int(self.b * 255), self.alpha
        return self.r, self.g, self.b, self.alpha

    def to_cmyk(self) -> tuple[float, float, float, float]:
        k = 1.0 - max(self.r, self.g, self.b)
        if k == 1.:
            return 0., 0., 0., 1.
        c = round((1. - self.r - k) / (1. - k), 10)
        m = round((1. - self.g - k) / (1. - k), 10)
        y = round((1. - self.b - k) / (1. - k), 10)
        return c, m, y, k

    def to_hsl(self) -> tuple[float, float, float]:
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        return round(h * 360., 6), round(s, 10), round(l, 10)

    def to_hex(self) -> str:
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'

    def to_hexa(self) -> str:
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}{self.alpha:02x}'

    def __repr__(self) -> str:
        h = self.to_hex()
        a = f', alpha={self.alpha:.3f}' if self.alpha < 1.0 else ''
        return f'{self.__class__.__name__}({h!r}{a})'