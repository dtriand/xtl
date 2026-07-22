from __future__ import annotations

import colorsys
from enum import Enum
from typing import Any

from colorspacious import cspace_convert
import numpy as np
from pydantic import field_validator, ValidationInfo, model_validator, model_serializer

from xtl.common.options import Option
from .base import BaseGraphModel
from xtl.graphs.base import BaseGraphModel


class Color(BaseGraphModel):
    """
    A representation of a color in RGB space.
    """

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
            desc='The opacity value of the color (0-1)',
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

    @model_validator(mode='before')
    @classmethod
    def _from_hex(cls, values):
        if isinstance(values, str):
            r, g, b, a = cls._parse_hex(values)
            return {'r': r, 'g': g, 'b': b, 'alpha': a}
        return values

    @model_serializer(when_used='json')
    def _to_hex(self) -> str:
        return self.to_hexa() if self.alpha < 1. else self.to_hex()

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

        :param value: A hex color string
        :return:
        """
        r, g, b, a = cls._parse_hex(value)
        return cls(r=r, g=g, b=b, alpha=a)

    @classmethod
    def from_name(cls, name: str) -> Color:
        """
        Construct a Color from a CSS4 named color (e.g., 'red', 'blue', 'green').

        :param name: A CSS4 color name
        :return:
        """
        import matplotlib.colors as mcolors

        if name not in mcolors.CSS4_COLORS:
            close = [k for k in mcolors.CSS4_COLORS if k.startswith(name[:3])]
            hint = f'  Did you mean one of: {",".join(close[:5])}' if close else ''
            raise ValueError(f'Unknown CSS color name {name!r}.{hint}\n'
                             f'See: https://www.w3.org/TR/css-color-4/#named-colors')
        hex_value: str = mcolors.CSS4_COLORS[name]
        return cls.from_hex(hex_value)

    @classmethod
    def from_cmyk(cls, c: float, m: float, y: float, k: float, alpha: float = 1.0) -> Color:
        """
        Construct a Color from a CMYK color space.

        :param c: The cyan value of the color (0-1)
        :param m: The magenta value of the color (0-1)
        :param y: The yellow value of the color (0-1)
        :param k: The key/black value of the color (0-1)
        :param alpha: The opacity value of the color (0-1)
        :return:
        """
        for value, channel in ((c, 'c'), (m, 'm'), (y, 'y'), (k, 'k')):
            if not 0 <= value <= 1:
                raise ValueError(f'CMYK channel {channel!r} must be between 0 and 1, got {value!r}')

        r = round((1. - c) * (1. - k), 10)
        g = round((1. - m) * (1. - k), 10)
        b = round((1. - c) * (1. - k), 10)
        return cls(r=r, g=g, b=b, alpha=alpha)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, alpha: float = 1.0) -> Color:
        """
        Construct a Color from an HSL color space.

        :param h: The hue value of the color (0-360)
        :param s: The saturation value of the color (0-1)
        :param l: The lightness value of the color (0-1)
        :param alpha: The opacity value of the color (0-1)
        :return:
        """
        if not (0. <= h <= 360.):
            raise ValueError(f'Hue must be between 0 and 360, got {h!r}')
        if not (0. <= s <= 1.):
            raise ValueError(f'Saturation must be between 0 and 1, got {s!r}')
        if not (0. <= l <= 1.):
            raise ValueError(f'Lightness must be between 0 and 1, got {l!r}')
        r, g, b = colorsys.hls_to_rgb(h / 360., l, s)
        return cls(r=round(r, 10), g=round(g, 10), b=round(b, 10), alpha=alpha)

    @classmethod
    def from_ucs(cls, Jp: float, ap: float, bp: float, a: float = 1.0) -> Color:
        """
        Construct a Color from a CAM02-UCS color space. The RGB values of the color are clipped between 0-1 to ensure
        valid RGB colors.

        :param Jp: The J' value of the color (lightness, typically 0-100)
        :param ap: The a' value of the color (red-green opponent axis, typically -120 to +120)
        :param bp: The b' value of the color (yellow-blue opponent axis, typically -120 to +120)
        :param a: The opacity value of the color (0-1)
        :return:
        """
        converted = cspace_convert((Jp, ap, bp), 'CAM02-UCS', 'sRGB1')
        # Clip values to avoid invalid RGB tuples
        r, g, b = np.clip(converted, 0, 1)
        return cls(r=round(r, 10), g=round(g, 10), b=round(b, 10), alpha=a)

    def to_rgb(self, as_int: bool = False) -> tuple[float, float, float]:
        """
        Convert to a RGB color tuple (r, g, b), where all values are between 0 and 1.

        :param as_int: If True, return RGB values as integers in the range 0-255. Default is False.
        :return: A tuple of (r, g, b) values in the RGB color space.
        """
        if as_int:
            return int(self.r * 255), int(self.g * 255), int(self.b * 255)
        return self.r, self.g, self.b

    def to_rgba(self, as_int: bool = False) -> tuple[float, float, float, float]:
        """
        Convert to a RGBA color tuple (r, g, b, a), where all values are between 0 and 1.

        :param as_int: If True, return RGB values as integers in the range 0-255, but A as a float in the range 0-1.
            Default is False.
        :return: A tuple of (r, g, b, a) values in the RGBA color space.
        """
        if as_int:
            r, g, b = self.to_rgb(as_int=True)
            return r, g, b, self.alpha
        return self.r, self.g, self.b, self.alpha

    def to_cmyk(self) -> tuple[float, float, float, float]:
        """
        Convert to a CMYK color tuple (c, m, y, k), where all values are bounded between 0 and 1.

        :return: A tuple of (c, m, y, k) values in the CMYK color space.
        """
        k = 1.0 - max(self.r, self.g, self.b)
        if k == 1.:
            return 0., 0., 0., 1.
        c = round((1. - self.r - k) / (1. - k), 10)
        m = round((1. - self.g - k) / (1. - k), 10)
        y = round((1. - self.b - k) / (1. - k), 10)
        return c, m, y, k

    def to_hsl(self) -> tuple[float, float, float]:
        """
        Convert to an HSL color tuple (h, s, l), where h is in degrees (between 0 and 360), and s, l are between 0 and
        1.

        :return: A tuple of (h, s, l) values in the HSL color space.
        """
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        return round(h * 360., 6), round(s, 10), round(l, 10)

    def to_ucs(self) -> tuple[float, float, float]:
        """
        Convert to a CAM02-UCS color tuple (J', a', b').

        :return: A tuple of (J', a', b') values in the CAM02-UCS color space.
        """
        # J': Lightness, a': red-green opponent axis, b': yellow-blue opponent axis
        Jp, ap, bp = cspace_convert(self.to_rgb(), 'sRGB1', 'CAM02-UCS')
        return float(Jp), float(ap), float(bp)

    def to_hex(self) -> str:
        """
        Convert to a hex color string.

        :return: A string representing the color in hex format (#RRGGBB).
        """
        r, g, b = self.to_rgb(as_int=True)
        return f'#{r:02x}{g:02x}{b:02x}'

    def to_hexa(self) -> str:
        """
        Convert to a hex color string including alpha.

        :return: A string representing the color in hex format (#RRGGBBAA).
        """
        r, g, b = self.to_rgb(as_int=True)
        return f'#{r:02x}{g:02x}{b:02x}{int(self.alpha * 255):02x}'

    def get_shades(self, n: int, *, Jp_range: tuple[float, float] = (30., 90.)) -> list[Color]:
        """
        Generate N perceptually equidistant shades of a base color, in terms of lightness. The interpolation is
        performed in CAM02-UCS color space. Note that the current color (base color) is not part of the shades, but
        rather only its a' and b' values are preserved.

        :param n: Number of shades to generate
        :param Jp_range: Range of lightness (J') values for the shades, as a tuple (min_J, max_J). Default is (30., 90.).
        :return: List of Color instances representing the shades.
        """
        # Convert base color to UCS0
        base_ucs = np.array(self.to_ucs())

        # Create shades equally spaced in J', but keeping a' and b' constant
        Jp_values = np.linspace(Jp_range[0], Jp_range[1], n)
        shades = np.array([[Jp, base_ucs[1], base_ucs[2]] for Jp in Jp_values])

        # Convert shades to RGB
        return [Color.from_ucs(*shade) for shade in shades]

    def __repr__(self) -> str:
        h = self.to_hex()
        a = f', alpha={self.alpha:.3f}' if self.alpha < 1.0 else ''
        return f'{self.__class__.__name__}({h!r}{a})'



COLORS: dict[str, tuple[Color, ...]] = {
    'pop': tuple(Color.from_hex(h) for h in ('#0372b1', '#ec672c', '#4eac47', '#cb2027', '#cf1e8a'))
}
"""
Named color lists for use in graphs.
"""

COLORS['default'] = COLORS['pop']
