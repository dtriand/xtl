from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt

from xtl.graphs.styles.stylesheets import StyleSource, StyleRegistry, StyleSheetMixin


STYLES = StyleRegistry(Path(__file__).parent / 'styles', pattern='*.mplstyle')


class MatplotlibStyleSheetMixin(StyleSheetMixin):

    @contextmanager
    def _apply_stylesheets(self, sources: list[StyleSource]) -> Generator[None, None, None]:
        resolved = [self._resolve_stylesheet_source(s) for s in sources]
        with plt.style.context(resolved):
            yield

    @staticmethod
    def _resolve_stylesheet_source(source: StyleSource) -> str | dict:
        if isinstance(source, dict):
            return source
        elif isinstance(source, str):
            try:
                path = STYLES[source]
                return str(path)
            except KeyError:
                if source in plt.style.available:
                    return source
                raise KeyError(f'Style {source!r} is neither built-in Matplotlib style nor a registered style.')
        elif isinstance(source, Path):
            resolved = source.expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f'Stylesheet file does not exist: {resolved}')
            return str(resolved)
        else:
            raise TypeError(f'Expected a style name, path or a dict of rc params; got {type(source).__name__}')
