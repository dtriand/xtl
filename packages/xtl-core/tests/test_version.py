from datetime import datetime

import pytest

from xtl_core.version import ReleaseLevel, VersionInfo


def _v(major=1, minor=2, micro=3, level=ReleaseLevel.FINAL, serial=0, date=None):
    return VersionInfo(major=major, minor=minor, micro=micro, level=level, serial=serial, date=date)


# level, release_level, string, pep440
LEVELS = [
    (ReleaseLevel.DEV, 'dev', '1.2.3dev4', '1.2.3.dev4'),
    (ReleaseLevel.ALPHA, 'a', '1.2.3a4', '1.2.3a4'),
    (ReleaseLevel.BETA, 'b', '1.2.3b4', '1.2.3b4'),
    (ReleaseLevel.GAMMA, 'g', '1.2.3g4', None),
    (ReleaseLevel.RC, 'rc', '1.2.3rc4', '1.2.3rc4'),
    (ReleaseLevel.FINAL, 'final', '1.2.3final4', '1.2.3.post4'),
]


class TestProperties:
    @pytest.mark.parametrize('level, release_level, string, pep440', LEVELS)
    def test_release_level(self, level, release_level, string, pep440):
        assert _v(level=level).release_level == release_level

    @pytest.mark.parametrize('level, release_level, string, pep440', LEVELS)
    def test_string(self, level, release_level, string, pep440):
        v = _v(level=level, serial=4)
        assert v.string == string
        assert str(v) == string

    def test_string_safe(self):
        assert _v(level=ReleaseLevel.ALPHA, serial=4).string_safe == '1.2.3'

    def test_tuple(self):
        v = _v(level=ReleaseLevel.BETA, serial=4)
        assert v.tuple == (1, 2, 3, 'b', 4)
        assert v.tuple_safe == (1, 2, 3)

    def test_release_date(self):
        assert _v().release_date == ''
        assert _v(date=datetime(2025, 6, 1)).release_date == '01/06/2025'

    def test_repr(self):
        v = _v(level=ReleaseLevel.ALPHA, serial=4, date=datetime(2025, 6, 1))
        assert repr(v) == "VersionInfo('1.2.3a4', hex='0x010203a4', date='01/06/2025')"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            _v().major = 2


class TestPep440:
    @pytest.mark.parametrize('level, release_level, string, pep440', LEVELS)
    def test_levels(self, level, release_level, string, pep440):
        v = _v(level=level, serial=4)
        if pep440 is None:
            with pytest.raises(ValueError, match='gamma'):
                _ = v.pep440
        else:
            assert v.pep440 == pep440

    def test_final_without_serial(self):
        assert _v(level=ReleaseLevel.FINAL, serial=0).pep440 == '1.2.3'

    def test_dev_without_serial(self):
        assert _v(level=ReleaseLevel.DEV, serial=0).pep440 == '1.2.3.dev0'


class TestHex:
    @pytest.mark.parametrize(
        'v, h',
        [
            (_v(level=ReleaseLevel.ALPHA, serial=4), '0x010203a4'),
            (_v(level=ReleaseLevel.FINAL, serial=0), '0x010203f0'),
            (_v(0, 2, 0, ReleaseLevel.DEV, 0), '0x00020000'),
            (_v(255, 255, 255, ReleaseLevel.RC, 15), '0xffffffdf'),
        ],
    )
    def test_hex(self, v, h):
        assert v.hex == h
        assert VersionInfo.from_hex(h) == v

    @pytest.mark.parametrize('level', list(ReleaseLevel))
    def test_round_trip(self, level):
        v = _v(level=level, serial=7)
        assert VersionInfo.from_hex(v.hex) == v

    def test_date(self):
        v = VersionInfo.from_hex('0x010203a4', '01/06/2025')
        assert v.date == datetime(2025, 6, 1)
        assert VersionInfo.from_hex('0x010203a4').date is None

    def test_invalid_level(self):
        # 0x1 is not a release level
        with pytest.raises(ValueError):
            VersionInfo.from_hex('0x01020314')


class TestFromStr:
    @pytest.mark.parametrize(
        'string, expected',
        [
            ('0.1.0', (0, 1, 0, ReleaseLevel.FINAL, 0)),
            ('1.2.10', (1, 2, 10, ReleaseLevel.FINAL, 0)),
            ('1.2.3final0', (1, 2, 3, ReleaseLevel.FINAL, 0)),
            ('1.2.3final2', (1, 2, 3, ReleaseLevel.FINAL, 2)),
            ('1.2.3a1', (1, 2, 3, ReleaseLevel.ALPHA, 1)),
            ('1.2.3b2', (1, 2, 3, ReleaseLevel.BETA, 2)),
            ('1.2.3g5', (1, 2, 3, ReleaseLevel.GAMMA, 5)),
            ('1.2.3rc3', (1, 2, 3, ReleaseLevel.RC, 3)),
            ('1.2.3dev4', (1, 2, 3, ReleaseLevel.DEV, 4)),
            ('1.2.13a1', (1, 2, 13, ReleaseLevel.ALPHA, 1)),
        ],
    )
    def test_valid(self, string, expected):
        v = VersionInfo.from_str(string)
        assert (v.major, v.minor, v.micro, v.level, v.serial) == expected
        assert v.date is None

    @pytest.mark.parametrize('string', ['0.1', '1.2.3.4', '1.2.a1', '1.2.3x1', '', 'a.b.c'])
    def test_invalid(self, string):
        with pytest.raises(ValueError):
            VersionInfo.from_str(string)

    @pytest.mark.parametrize('level', list(ReleaseLevel))
    def test_round_trip(self, level):
        v = _v(level=level, serial=4)
        assert VersionInfo.from_str(v.string) == v

    def test_date(self):
        v = VersionInfo.from_str('1.2.3', '01/06/2025')
        assert v.date == datetime(2025, 6, 1)
        with pytest.raises(ValueError):
            VersionInfo.from_str('1.2.3', '2025-06-01')


class TestFromValue:
    def test_instance(self):
        v = _v()
        assert VersionInfo.from_value(v) is v

    def test_dict(self):
        assert (
            VersionInfo.from_value({'major': 1, 'minor': 2, 'micro': 3, 'level': ReleaseLevel.FINAL, 'serial': 0})
            == _v()
        )

    def test_hex_string(self):
        assert VersionInfo.from_value('0x010203a4') == _v(level=ReleaseLevel.ALPHA, serial=4)

    def test_version_string(self):
        assert VersionInfo.from_value('1.2.3a4') == _v(level=ReleaseLevel.ALPHA, serial=4)

    def test_invalid_string(self):
        # Valid hex, but not a valid version: falls back to from_str, which fails too
        with pytest.raises(ValueError):
            VersionInfo.from_value('10')

    @pytest.mark.parametrize('value', [1, 1.2, None, ['1', '2', '3']])
    def test_invalid_type(self, value):
        with pytest.raises(TypeError):
            VersionInfo.from_value(value)
