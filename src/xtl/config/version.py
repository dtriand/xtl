from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


class ReleaseLevel(IntEnum):
    """
    Enum to represent the version release level.
    """
    DEV = 0     # 0x0 in hex
    """Development version, not released yet."""
    ALPHA = 10  # 0xa
    """Alpha version, early testing phase."""
    BETA = 11   # 0xb
    """Beta version, feature complete but may have bugs."""
    GAMMA = 12  # 0xc
    """Gamma version, stable but not final."""
    RC = 13     # 0xd
    """Release Candidate, ready for final testing."""
    FINAL = 15  # 0xf
    """Final version, ready for production use."""


@dataclass
class VersionInfo:
    """
    Dataclass to hold version information.
    """

    major: int
    """Major version number, incremented for incompatible changes."""
    minor: int
    """Minor version number, incremented for new features."""
    micro: int
    """Micro version number, incremented for bug fixes."""
    level: ReleaseLevel
    """Release level of the version, indicating the stability and readiness for release."""
    serial: int
    """Serial number for the release level, used to differentiate between multiple releases of the same level."""
    date: Optional[datetime] = None
    """Release date of the version, if available. This is optional and can be None."""

    @property
    def release_level(self) -> str:
        """
        Return the release level as a string.
        """
        rl = self.level.name.lower()
        if rl in ['alpha', 'beta', 'gamma']:
            rl = rl[0]
        return rl

    @property
    def release_date(self) -> str:
        """
        Return the release date as a string.
        """
        if self.date is None:
            return ''
        return self.date.strftime('%d/%m/%Y')

    @property
    def string(self):
        """
        Return a string representation of the version (e.g., '1.2.3a4').
        """
        return (f'{self.major}.{self.minor}.{self.micro}{self.release_level}'
                f'{self.serial}')

    @property
    def string_safe(self):
        """
        Return a string representation of the version including only major, minor and
        micro levels (e.g., '1.2.3').
        """
        return f'{self.major}.{self.minor}.{self.micro}'

    @property
    def tuple(self) -> Tuple[int, int, int, str, int]:
        """
        Return a tuple representation of the version (e.g., (1, 2, 3, 'a', 4)).
        """
        return self.major, self.minor, self.micro, self.release_level, self.serial

    @property
    def tuple_safe(self) -> Tuple[int, int, int]:
        """
        Return a tuple representation of the version including only major, minor and
        micro levels (e.g., (1, 2, 3)).
        """
        return self.major, self.minor, self.micro

    @property
    def hex(self) -> str:
        """
        Return a 32-bit hexadecimal representation of the version (e.g., '0x0102030a4').
        """
        h = int(self.serial)
        h |= self.level.value * 1 << 4
        h |= int(self.micro) * 1 << 8
        h |= int(self.minor) * 1 << 16
        h |= int(self.major) * 1 << 24
        return f'0x{h:08x}'

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}("{self.string}", hex="{self.hex}", '
                f'date="{self.release_date}")')


def version_from_str(version_str: str, date_str: str = None) -> VersionInfo:
    """
    Create a VersionInfo object from a version string.
    """
    parts = version_str.split('.')
    major = int(parts[0])
    minor = int(parts[1])

    i = 0
    for i in range(len(parts[2])):
        if not parts[2][i].isdigit():
            break
    if i == 0:
        raise ValueError('Invalid version string. '
                         'Expected {MAJOR}.{MINOR}.{MICRO}{LEVEL}{SERIAL} but did not '
                         'find {LEVEL}')

    micro = int(parts[2][:i])

    level_bit = parts[2][i]
    if level_bit == 'a':
        level, serial = ReleaseLevel.ALPHA, int(parts[2][i + 1:])
    elif level_bit == 'b':
        level, serial = ReleaseLevel.BETA, int(parts[2][i + 1:])
    elif level_bit == 'g':
        level, serial = ReleaseLevel.GAMMA, int(parts[2][i + 1:])
    elif level_bit == 'r':
        level, serial = ReleaseLevel.RC, int(parts[2][i + 2:])
    elif level_bit == 'f':
        level, serial = ReleaseLevel.FINAL, int(parts[2][i + 5:])
    elif level_bit == 'd':
        level, serial = ReleaseLevel.DEV, int(parts[2][i + 3:])
    else:
        raise ValueError(f'Invalid version string. Unknown level: {level_bit}')

    if date_str is not None:
        date = datetime.strptime(date_str, '%d/%m/%Y')
    else:
        date = None
    return VersionInfo(major=major, minor=minor, micro=micro, level=level,
                       serial=serial, date=date)


def version_from_hex(hex_str: str, date_str: str = None) -> VersionInfo:
    """
    Create a VersionInfo object from a 32-bit hexadecimal string.
    """
    h = int(hex_str, 16)
    major = (h >> 24) & 0xff
    minor = (h >> 16) & 0xff
    micro = (h >> 8) & 0xff
    level = ReleaseLevel((h >> 4) & 0xf)
    serial = h & 0xf
    if date_str is not None:
        date = datetime.strptime(date_str, '%d/%m/%Y')
    else:
        date = None
    return VersionInfo(major=major, minor=minor, micro=micro, level=level,
                       serial=serial, date=date)


# Unique place for version definition
version = VersionInfo(
    major=0,
    minor=2,
    micro=0,
    level=ReleaseLevel.DEV,
    serial=0,  # < 16
    date=datetime(year=2025, month=6, day=1),
)
"""XTL version information"""
