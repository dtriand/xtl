from typing import Any


def deepmerge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, with values from the `override` dictionary taking precedence over those in the
    `base`.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deepmerge(out[k], v)
        else:
            out[k] = v
    return out


def find_unpicklable(obj: Any, path: str | None = None) -> Any:
    """
    Recursively search for the first unpicklable object within a nested structure, returning the offending object if
    found or `None` if all objects are picklable.
    """
    import pickle

    if path is None:
        path = type(obj).__name__

    try:
        pickle.dumps(obj)
        return None
    except Exception:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            result = find_unpicklable(v, f'{path}[{k!r}]')
            if result:
                return result
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            result = find_unpicklable(v, f'{path}[{i}]')
            if result:
                return result

    return path, type(obj), repr(obj)


def is_picklable(obj: Any, path: str | None = None) -> tuple[bool, tuple[str, type, str] | None]:
    """
    Check if an object is picklable, returning a tuple of (is_picklable, unpicklable_info) where
    `is_picklable` is a boolean indicating whether the object can be pickled and
    `unpicklable_info` is either `None` if the object is picklable or a tuple of (path, type, repr) describing the
    first unpicklable object found within the structure.
    """
    bad = find_unpicklable(obj, path)
    if bad is not None:
        return False, bad
    return True, None
