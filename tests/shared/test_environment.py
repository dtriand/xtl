from pathlib import Path

from xtl.shared.environment import XTL_COMPUTE_SITE, XTL_CONFIG, XTL_FONTS_DIR


def test_names():
    assert (XTL_CONFIG.name, XTL_COMPUTE_SITE.name, XTL_FONTS_DIR.name) == (
        'XTL_CONFIG',
        'XTL_COMPUTE_SITE',
        'XTL_FONTS_DIR',
    )


def test_paths_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv('XTL_TEST_DIR', str(tmp_path))
    monkeypatch.setenv('XTL_CONFIG', '$XTL_TEST_DIR/xtl.toml')
    assert XTL_CONFIG.value == tmp_path / 'xtl.toml'
    monkeypatch.setenv('XTL_FONTS_DIR', '~/fonts')
    assert XTL_FONTS_DIR.value == Path.home() / 'fonts'
