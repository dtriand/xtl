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
