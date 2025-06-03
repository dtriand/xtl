"""
Serializers of gemmi objects for use with `Pydantic` models.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gemmi


def GemmiUnitCell(x: 'gemmi.UnitCell') -> tuple[float, ...]:
    """
    Serializes a `gemmi.UnitCell` instance to a tuple.

    :param x: `gemmi.UnitCell` instance
    :return: Tuple representation of the unit cell
    """
    return x.a, x.b, x.c, x.alpha, x.beta, x.gamma


def GemmiSpaceGroup(x: 'gemmi.SpaceGroup') -> str:
    """
    Serializes a `gemmi.SpaceGroup` instance to its Hermann-Mauguin notation.

    :param x: `gemmi.SpaceGroup` instance
    :return: Space group symbol
    """
    return x.hm


def GemmiMat33(x: 'gemmi.Mat33') -> tuple[tuple[float, ...], ...]:
    """
    Serializes a `gemmi.Mat33` instance to a tuple of tuples.

    :param x: `gemmi.Mat33` instance
    :return: List of lists representation of the matrix
    """
    return tuple(tuple(row) for row in x.tolist())
