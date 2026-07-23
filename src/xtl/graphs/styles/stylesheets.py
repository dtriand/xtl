from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator

StyleSource = Path | dict[str, Any] | str


class StyleSheetMixin(ABC):

    def style(self, *sources: StyleSource) -> AbstractContextManager:
        return self._style_context(list(sources))

    @contextmanager
    def _style_context(self, sources: list[StyleSource | str]) -> Generator[None, None, None]:
        with self._apply_stylesheets(sources):
            yield

    @abstractmethod
    @contextmanager
    def _apply_stylesheets(self, sources: list[StyleSource | str]) -> Generator[None, None, None]:
        raise NotImplementedError()


class StyleRegistry:

    def __init__(self, styles_dir: Path | str, pattern: str | None = None) -> None:
        self._styles: dict[str, Path] = {}
        self._discover_styles(Path(styles_dir), pattern)

    def _discover_styles(self, styles_dir: Path, pattern: str | None) -> None:
        if not styles_dir.exists():
            raise FileNotFoundError(f'Path does not exist: {styles_dir}')
        if not styles_dir.is_dir():
            raise NotADirectoryError(f'Path is not a directory: {styles_dir}')

        pattern = pattern or '*'
        for path in sorted(styles_dir.glob(pattern)):
            self._styles[path.stem] = path.resolve()

    def __getitem__(self, key: str) -> Path:
        try:
            return self._styles[key]
        except KeyError:
            raise KeyError(f'No built-in style with name {key!r}. '
                           f'Available styles: {", ".join(self._styles.keys())}') from None

    def __contains__(self, key: str) -> bool:
        return key in self._styles

    def __iter__(self) -> Iterator[str]:
        return iter(self._styles)

    def __len__(self) -> int:
        return len(self._styles)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({{{", ".join(f"{k!r}" for k in sorted(self._styles))}}})'

    def get(self, key: str, default: Path | None = None) -> Path | None:
        return self._styles.get(key, default)

    def keys(self):
        return self._styles.keys()

    def values(self):
        return self._styles.values()

    def items(self):
        return self._styles.items()

