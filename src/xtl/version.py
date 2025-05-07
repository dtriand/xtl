from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


class ReleaseLevel(IntEnum):
    """
    Enum to represent the version release level.
    """
    DEV = 0     # 0x0 in hex
    ALPHA = 10  # 0xa
    BETA = 11   # 0xb
    GAMMA = 12  # 0xc
    RC = 13     # 0xd
    FINAL = 15  # 0xf


@dataclass
class VersionInfo:
    """
    Dataclass to hold version information.
    """

    major: int
    minor: int
    micro: int
    level: ReleaseLevel
    serial: int
    date: Optional[datetime] = None

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
    def tuple(self) -> Tuple[int, int, int, str, int]:
        """
        Return a tuple representation of the version (e.g., (1, 2, 3, 'a', 4)).
        """
        return self.major, self.minor, self.micro, self.release_level, self.serial

    @property
    def safe_tuple(self) -> Tuple[int, int, int]:
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

    if parts[2][i] == 'a':
        level, serial = ReleaseLevel.ALPHA, int(parts[2][i + 1:])
    elif parts[2][i] == 'b':
        level, serial = ReleaseLevel.BETA, int(parts[2][i + 1:])
    elif parts[2][i] == 'g':
        level, serial = ReleaseLevel.GAMMA, int(parts[2][i + 1:])
    elif parts[2][i] == 'r':
        level, serial = ReleaseLevel.RC, int(parts[2][i + 2:])
    elif parts[2][i] == 'f':
        level, serial = ReleaseLevel.FINAL, int(parts[2][i + 5:])
    elif parts[2][i] == 'd':
        level, serial = ReleaseLevel.DEV, int(parts[2][i + 3:])
    else:
        raise ValueError(f'Invalid version string. Unknown level: {parts[2][i]}')

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
    minor=1,
    micro=0,
    level=ReleaseLevel.ALPHA,
    serial=0,  # < 16
    date=datetime(year=2025, month=6, day=1),
)
"""XTL version information"""
