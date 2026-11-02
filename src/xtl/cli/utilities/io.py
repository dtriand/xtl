import glob
from pathlib import Path
from typing import Any

import typer


def resolve_paths_or_glob(raw: list[Any]) -> list[Path]:
    """
    Resolve a list of literal paths or glob patterns to a flat list of Paths.

    For each entry:
      - If it resolves to an existing path, use it as-is.
      - Otherwise, treat it as a glob pattern and expand it.

    Raises typer.BadParameter if a pattern matches nothing.
    """
    resolved: list[Path] = []
    for entry in raw:
        p = Path(entry)
        if p.exists():
            resolved.append(p)
        else:
            matches = glob.glob(str(entry))
            if not matches:
                raise typer.BadParameter(
                    f"'{entry}' is neither an existing file nor a glob pattern with matches"
                )
            resolved.extend(map(Path, matches))
    return resolved

